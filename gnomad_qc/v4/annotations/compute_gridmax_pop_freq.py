"""
Compute GridMax companion frequencies for a leakage-free population AF comparison.

Companion to compute_gridmax_freq.py: same gnomAD v4 exome samples, same bin
assignments and train/eval split (read from the existing gridmax training samples
HT). Two experiments:

  Experiment 1: adj AC/AN per genetic ancestry group computed on the training fold
  only (also on the eval fold and on all samples, for validation). The high-AB-het
  correction follows the release logic: applied to every group of a sample set when
  the site AF of that sample set is above --af-threshold.

  Experiment 2: adj AC/AN per bin pooling both folds, released only for bins with at
  least --min-bin-occupancy exome samples. Derived from the per-fold per-bin call
  stats of compute_gridmax_freq.py --compute-freq (uncorrected + high-AB-het counts),
  with the high-AB-het correction re-applied on the pooled per-bin AF.

Pipeline steps (run all by default; use flags to run individual steps):
  --prep-vds                 Filter VDS to binned exome samples (both folds), keep pop, split multiallelics
  --densify                  Densify the split VDS once and checkpoint the dense MT to --tmp-dir
  --compute-bin-freq         Per-bin, per-fold call stats (as compute_gridmax_freq.py --compute-freq);
                             not needed with --use-production-bin-freq
  --compute-pop-freq         Per-group call stats for train / eval / all samples; with
                             --pooled-bins-direct, also per-bin call stats on pooled samples
  --compute-pooled-bin-freq  Pool per-fold per-bin call stats and suppress small bins
  --validate                 Aggregate-only consistency checks, written to a JSON file
  --export                   Export Experiment 1 and Experiment 2 TSVs

All outputs are written under --output-dir. Production gnomAD inputs are only read.
Per-bin intermediates (temp/ and the direct pooled per-bin field) are not
privacy-filtered.
"""

import argparse
import json
import logging
from typing import Dict, List, Optional

import hail as hl

from gnomad_qc.v4.annotations.compute_gridmax_freq import (
    DEFAULT_AB_CUTOFF,
    DEFAULT_AF_THRESHOLD,
    DEFAULT_MIN_BIN_OCCUPANCY,
    aggregate_pixel_freq,
    prep_vds_for_gridmax,
)
from gnomad_qc.v4.annotations.generate_freq import densify_and_prep_vds_for_freq
from gnomad_qc.v4.resources.annotations import (
    get_gridmax_freq,
    get_gridmax_training_samples,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s: %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
)
logger = logging.getLogger("gridmax_pop_freq")
logger.setLevel(logging.INFO)

PUBLIC_EXOMES_RELEASE_HT = (
    "gs://gcp-public-data--gnomad/release/4.1/ht/exomes/gnomad.exomes.v4.1.sites.ht"
)
TEST_GENE_INTERVAL = "chr11:113409605-113475691"
SAMPLE_SETS = {"train": [0], "eval": [1], "all": [0, 1]}
ALL_GROUP = "all"
RAW_FIELDS = ["AC", "AN", "homozygote_count", "n_high_ab_het"]


def get_output_paths(
    output_dir: str,
    tmp_dir: str,
    chrom: Optional[str] = None,
    test_gene: bool = False,
) -> Dict[str, str]:
    """
    Get the output paths of every step.

    :param output_dir: Root output directory.
    :param tmp_dir: Temporary directory (dense MT checkpoint).
    :param chrom: Optional chromosome suffix.
    :param test_gene: Whether to use the test-gene (DRD2) suffix.
    :return: Dict of output paths keyed by output name.
    """
    suffix = (".test_gene" if test_gene else "") + (f".{chrom}" if chrom else "")
    out = output_dir.rstrip("/")
    tmp = tmp_dir.rstrip("/")
    return {
        "split_vds": f"{out}/temp/gridmax.split_vds{suffix}.vds",
        "dense_mt": f"{tmp}/gridmax_pop_freq.dense{suffix}.mt",
        "bin_freq": f"{out}/temp/gridmax.freq{suffix}.ht",
        "pop_freq": f"{out}/gridmax.pop_freq{suffix}.ht",
        "pooled_bin_freq": f"{out}/gridmax.pooled_bin_freq{suffix}.ht",
        "validation": f"{out}/gridmax.validation{suffix}.json",
        "train_pop_tsv": f"{out}/gridmax.train_fold_pop_ac_an{suffix}.tsv.bgz",
        "pooled_bin_tsv": f"{out}/gridmax.pooled_bin_ac_an{suffix}.tsv.bgz",
    }


def get_scope_intervals(
    chrom: Optional[str] = None, test_gene: bool = False
) -> Optional[List[hl.expr.IntervalExpression]]:
    """
    Get the genomic intervals covered by the run.

    :param chrom: Optional chromosome.
    :param test_gene: Whether the run is restricted to DRD2.
    :return: List of intervals, or None for a genome-wide run.
    """
    if test_gene:
        return [hl.parse_locus_interval(TEST_GENE_INTERVAL, reference_genome="GRCh38")]
    if chrom:
        return [hl.parse_locus_interval(chrom, reference_genome="GRCh38")]
    return None


def _af(ac: hl.expr.Int32Expression, an: hl.expr.Int32Expression):
    return hl.if_else(an > 0, hl.float64(ac) / an, hl.missing(hl.tfloat64))


def _zero_raw() -> hl.expr.StructExpression:
    return hl.struct(**{f: hl.int32(0) for f in RAW_FIELDS})


def _sum_raw(stats_array: hl.expr.ArrayExpression) -> hl.expr.StructExpression:
    return stats_array.fold(
        lambda acc, s: hl.struct(**{f: acc[f] + s[f] for f in RAW_FIELDS}),
        _zero_raw(),
    )


def _flatten_raw(stats: hl.expr.StructExpression) -> hl.expr.StructExpression:
    """Reduce a struct(call_stats, n_high_ab_het) to alt-only raw scalars."""
    return hl.struct(
        AC=stats.call_stats.AC[1],
        AN=stats.call_stats.AN,
        homozygote_count=stats.call_stats.homozygote_count[1],
        n_high_ab_het=stats.n_high_ab_het,
    )


def _corrected_call_stats(
    raw: hl.expr.StructExpression, apply_correction: hl.expr.BooleanExpression
) -> hl.expr.StructExpression:
    """
    Apply the high-AB-het correction to raw call stats when requested.

    Reclassifying n high-AB het calls as hom-alt adds n alt alleles (AN unchanged)
    and n hom-alt samples.
    """
    n = hl.if_else(hl.coalesce(apply_correction, False), raw.n_high_ab_het, 0)
    ac = raw.AC + n
    return hl.struct(
        AC=ac,
        AF=_af(ac, raw.AN),
        AN=raw.AN,
        homozygote_count=raw.homozygote_count + n,
    )


def _bin_corrected(
    raw: hl.expr.StructExpression, af_threshold: float
) -> hl.expr.StructExpression:
    """Correct per-bin call stats using the bin's own AF (compute_gridmax_freq.py logic)."""
    return _corrected_call_stats(raw, _af(raw.AC, raw.AN) > af_threshold)


def densify_split_vds(vds: hl.vds.VariantDataset, ab_cutoff: float) -> hl.MatrixTable:
    """
    Densify the split VDS and keep only the entries needed for call stats.

    :param vds: Split VDS from --prep-vds (with pop, bin_id, eval_fold cols).
    :param ab_cutoff: Allele balance cutoff for the high-AB-het annotation.
    :return: Dense MatrixTable with GT, adj and _high_ab_het_ref entries.
    """
    mt = densify_and_prep_vds_for_freq(vds, ab_cutoff=ab_cutoff)
    return mt.select_entries("GT", "adj", "_high_ab_het_ref")


def compute_pop_freq(
    mt: hl.MatrixTable,
    pops: List[str],
    af_threshold: float = DEFAULT_AF_THRESHOLD,
    pooled_bins_direct: bool = False,
) -> hl.Table:
    """
    Compute adj call stats per genetic ancestry group for train / eval / all samples.

    One group_by over (pop, eval_fold) is aggregated, then summed into the three
    sample sets. For each set, the high-AB-het correction is applied to every group
    when the set's overall site AF is above `af_threshold` (release logic).

    :param mt: Dense MatrixTable with pop, eval_fold, bin_id and fixed_homalt_model cols.
    :param pops: Genetic ancestry groups to report.
    :param af_threshold: Site AF above which the high-AB-het correction is applied.
    :param pooled_bins_direct: Also compute per-bin call stats on pooled samples
        (freq_pooled_direct; per-bin AF correction), to validate Experiment 2.
    :return: Table with raw_{set} and freq_{set} structs (one field per group plus
        'all') and optionally freq_pooled_direct.
    """
    high_ab_het_flag = mt.GT.is_het_ref() & ~mt.fixed_homalt_model & mt._high_ab_het_ref
    stats_agg = hl.struct(
        call_stats=hl.agg.call_stats(mt.GT, mt.alleles),
        n_high_ab_het=hl.int32(hl.agg.count_where(high_ab_het_flag)),
    )
    row_aggs = {
        "pop_fold_stats": hl.agg.filter(
            mt.adj,
            hl.agg.group_by(hl.struct(pop=mt.pop, eval_fold=mt.eval_fold), stats_agg),
        )
    }
    if pooled_bins_direct:
        row_aggs["bin_stats_pooled_direct"] = hl.agg.filter(
            mt.adj, hl.agg.group_by(mt.bin_id, stats_agg)
        )
    logger.info(
        "Aggregating per-pop call stats (and pooled per-bin: %s)...", pooled_bins_direct
    )
    ht = mt.select_rows(**row_aggs).rows()

    def _raw_for(pop: str, folds: List[int]) -> hl.expr.StructExpression:
        items = hl.array(
            [
                ht.pop_fold_stats.get(hl.struct(pop=pop, eval_fold=hl.int32(f)))
                for f in folds
            ]
        )
        return _sum_raw(items.filter(hl.is_defined).map(_flatten_raw))

    raw_annotations = {}
    for set_name, folds in SAMPLE_SETS.items():
        raw = {pop: _raw_for(pop, folds) for pop in pops}
        raw[ALL_GROUP] = _sum_raw(hl.array(list(raw.values())))
        raw_annotations[f"raw_{set_name}"] = hl.struct(**raw)
    ht = ht.annotate(**raw_annotations)

    groups = pops + [ALL_GROUP]
    ht = ht.annotate(
        **{
            f"freq_{set_name}": hl.bind(
                lambda r: hl.struct(
                    **{
                        g: _corrected_call_stats(
                            r[g], _af(r[ALL_GROUP].AC, r[ALL_GROUP].AN) > af_threshold
                        )
                        for g in groups
                    }
                ),
                ht[f"raw_{set_name}"],
            )
            for set_name in SAMPLE_SETS
        }
    )
    if pooled_bins_direct:
        ht = ht.annotate(
            freq_pooled_direct=hl.dict(
                ht.bin_stats_pooled_direct.items().map(
                    lambda kv: (
                        kv[0],
                        _bin_corrected(_flatten_raw(kv[1]), af_threshold),
                    )
                )
            )
        )
    ht = ht.drop(
        "pop_fold_stats", *(["bin_stats_pooled_direct"] if pooled_bins_direct else [])
    )
    return ht.select_globals(af_threshold_for_correction=af_threshold, pops=pops)


def pool_bin_folds(freq_ht: hl.Table, af_threshold: float) -> hl.expr.DictExpression:
    """
    Pool per-fold per-bin call stats and re-apply the high-AB-het correction.

    :param freq_ht: Output of compute_gridmax_freq.py --compute-freq (uncorrected
        freq / freq_eval dicts and high_ab_hets / high_ab_hets_eval counts by bin_id).
    :param af_threshold: Per-bin pooled AF above which the correction is applied.
    :return: Dict bin_id -> pooled corrected call stats (no privacy filter).
    """
    zero_cs = hl.struct(
        AC=hl.int32(0),
        AF=hl.missing(hl.tfloat64),
        AN=hl.int32(0),
        homozygote_count=hl.int32(0),
    )
    bins = hl.set(freq_ht.freq.keys()).union(hl.set(freq_ht.freq_eval.keys()))

    def _pooled(b):
        tr = hl.coalesce(freq_ht.freq.get(b), zero_cs)
        ev = hl.coalesce(freq_ht.freq_eval.get(b), zero_cs)
        n = hl.coalesce(freq_ht.high_ab_hets.get(b), hl.int64(0)) + hl.coalesce(
            freq_ht.high_ab_hets_eval.get(b), hl.int64(0)
        )
        raw = hl.struct(
            AC=tr.AC + ev.AC,
            AN=tr.AN + ev.AN,
            homozygote_count=tr.homozygote_count + ev.homozygote_count,
            n_high_ab_het=hl.int32(n),
        )
        return _bin_corrected(raw, af_threshold)

    return hl.dict(hl.array(bins).map(lambda b: (b, _pooled(b))))


def get_pooled_bin_counts(training_ht: hl.Table) -> Dict[str, int]:
    """
    Count exome samples per bin, pooling both folds.

    :param training_ht: GridMax training samples HT.
    :return: Dict bin_id -> number of exome samples.
    """
    ht = training_ht.filter(
        hl.is_defined(training_ht.bin_id) & (training_ht.data_type == "exome")
    )
    return ht.aggregate(hl.agg.counter(ht.bin_id))


def compute_pooled_bin_freq(
    freq_ht: hl.Table,
    pooled_counts: Dict[str, int],
    af_threshold: float = DEFAULT_AF_THRESHOLD,
    min_bin_occupancy: int = DEFAULT_MIN_BIN_OCCUPANCY,
) -> hl.Table:
    """
    Compute privacy-filtered per-bin call stats pooling both folds.

    :param freq_ht: Output of compute_gridmax_freq.py --compute-freq.
    :param pooled_counts: Dict bin_id -> pooled exome sample count.
    :param af_threshold: Per-bin pooled AF above which the correction is applied.
    :param min_bin_occupancy: Bins with fewer pooled exome samples are suppressed.
    :return: Table with freq_pooled dict (released bins, AN > 0 only).
    """
    released = {b: n for b, n in pooled_counts.items() if n >= min_bin_occupancy}
    logger.info(
        "%d bins pass min_bin_occupancy=%d (pooled exome counts).",
        len(released),
        min_bin_occupancy,
    )
    released_set = hl.literal(set(released), hl.tset(hl.tstr))
    pooled = pool_bin_folds(freq_ht, af_threshold)
    ht = freq_ht.select(
        freq_pooled=hl.dict(
            pooled.items().filter(
                lambda kv: released_set.contains(kv[0]) & (kv[1].AN > 0)
            )
        )
    )
    return ht.select_globals(
        af_threshold_for_correction=af_threshold,
        min_bin_occupancy=min_bin_occupancy,
        sample_counts_pooled=hl.literal(released, hl.tdict(hl.tstr, hl.tint64)),
    )


def _public_group_indices(
    freq_meta: List[Dict[str, str]], pops: List[str]
) -> Dict[str, int]:
    idx = {}
    for i, meta in enumerate(freq_meta):
        if meta.get("group") != "adj":
            continue
        if len(meta) == 1:
            idx[ALL_GROUP] = i
        elif len(meta) == 2:
            pop = meta.get("gen_anc", meta.get("pop"))
            if pop in pops:
                idx[pop] = i
    return idx


def validate(
    pop_ht: hl.Table,
    bin_freq_ht: hl.Table,
    pooled_ht: hl.Table,
    pooled_counts: Dict[str, int],
    pops: List[str],
    af_threshold: float,
    min_bin_occupancy: int,
    intervals: Optional[list] = None,
) -> dict:
    """
    Run aggregate-only consistency checks.

    :param pop_ht: Output of --compute-pop-freq.
    :param bin_freq_ht: Per-fold per-bin call stats (compute_gridmax_freq.py format).
    :param pooled_ht: Output of --compute-pooled-bin-freq.
    :param pooled_counts: Dict bin_id -> pooled exome sample count.
    :param pops: Genetic ancestry groups.
    :param af_threshold: AF threshold used for the high-AB-het correction.
    :param min_bin_occupancy: Privacy threshold.
    :param intervals: Optional intervals to restrict the public release HT.
    :return: Dict of aggregate results (counts, fractions, max differences only).
    """
    results = {}
    groups = pops + [ALL_GROUP]

    logger.info("Check 1: all-sample per-group AC/AN vs public v4.1 exomes...")
    pub = hl.read_table(PUBLIC_EXOMES_RELEASE_HT)
    if intervals is not None:
        pub = hl.filter_intervals(pub, intervals)
    pub_idx = _public_group_indices(hl.eval(pub.freq_meta), pops)
    results["public_groups_found"] = sorted(pub_idx)
    pub = pub.select(pub=hl.struct(**{g: pub.freq[i] for g, i in pub_idx.items()}))
    ht = pop_ht.annotate(pub=pub[pop_ht.key].pub)
    in_scope = ht.freq_all[ALL_GROUP].AN > 0
    per_group = {}
    for g in pub_idx:
        ours = ht.freq_all[g]
        theirs = ht.pub[g]
        both = in_scope & hl.is_defined(theirs)
        per_group[g] = hl.agg.filter(
            both,
            hl.struct(
                n_compared=hl.agg.count(),
                n_ac_exact=hl.agg.count_where(ours.AC == theirs.AC),
                n_an_exact=hl.agg.count_where(ours.AN == theirs.AN),
                max_abs_ac_diff=hl.agg.max(hl.abs(ours.AC - theirs.AC)),
                max_abs_an_diff=hl.agg.max(hl.abs(ours.AN - theirs.AN)),
                max_abs_af_diff=hl.agg.max(
                    hl.abs(hl.coalesce(ours.AF, 0.0) - hl.coalesce(theirs.AF, 0.0))
                ),
            ),
        )
    agg = ht.aggregate(
        hl.struct(
            n_variants=hl.agg.count(),
            n_variants_an_gt0=hl.agg.count_where(in_scope),
            n_ac_gt0_missing_in_public=hl.agg.count_where(
                in_scope & (ht.freq_all[ALL_GROUP].AC > 0) & hl.is_missing(ht.pub)
            ),
            per_group=hl.struct(**per_group),
        )
    )
    results["check1_vs_public"] = _to_py(agg)

    logger.info("Check 2: per-pop totals vs per-bin totals (raw, per fold)...")
    ht = pop_ht.annotate(bins=bin_freq_ht[pop_ht.key])

    def _bin_total(d):
        return hl.struct(
            AC=hl.sum(d.values().map(lambda s: s.AC)),
            AN=hl.sum(d.values().map(lambda s: s.AN)),
        )

    tr_bins = _bin_total(ht.bins.freq)
    ev_bins = _bin_total(ht.bins.freq_eval)
    agg = ht.aggregate(
        hl.struct(
            n_variants=hl.agg.count(),
            n_missing_in_bin_ht=hl.agg.count_where(hl.is_missing(ht.bins)),
            n_train_mismatch=hl.agg.count_where(
                (tr_bins.AC != ht.raw_train[ALL_GROUP].AC)
                | (tr_bins.AN != ht.raw_train[ALL_GROUP].AN)
            ),
            n_eval_mismatch=hl.agg.count_where(
                (ev_bins.AC != ht.raw_eval[ALL_GROUP].AC)
                | (ev_bins.AN != ht.raw_eval[ALL_GROUP].AN)
            ),
            n_train_plus_eval_ne_all=hl.agg.count_where(
                hl.any(
                    [
                        (ht.raw_train[g][f] + ht.raw_eval[g][f]) != ht.raw_all[g][f]
                        for g in groups
                        for f in RAW_FIELDS
                    ]
                )
            ),
            n_ac_gt_an=hl.agg.count_where(
                hl.any(
                    [
                        ht[f"freq_{s}"][g].AC > ht[f"freq_{s}"][g].AN
                        for s in SAMPLE_SETS
                        for g in groups
                    ]
                )
            ),
        )
    )
    results["check2_pop_vs_bins"] = _to_py(agg)

    if "freq_pooled_direct" in pop_ht.row:
        logger.info("Check 3: pooled per-bin stats, summed folds vs direct...")
        ht = pop_ht.annotate(bins=bin_freq_ht[pop_ht.key])
        derived = pool_bin_folds(ht.bins, af_threshold)
        direct = ht.freq_pooled_direct
        agg = ht.aggregate(
            hl.struct(
                n_variants=hl.agg.count(),
                n_entries_direct=hl.agg.sum(hl.len(direct)),
                n_entries_derived=hl.agg.sum(hl.len(derived)),
                n_variants_mismatch=hl.agg.count_where(derived != direct),
            )
        )
        results["check3_pooled_derived_vs_direct"] = _to_py(agg)

    logger.info("Check 4: privacy and arithmetic of the released pooled per-bin HT...")
    released = {b for b, n in pooled_counts.items() if n >= min_bin_occupancy}
    released_lit = hl.literal(released, hl.tset(hl.tstr))
    entries = pooled_ht.freq_pooled.items()
    agg = pooled_ht.aggregate(
        hl.struct(
            n_variants=hl.agg.count(),
            n_entries=hl.agg.sum(hl.len(entries)),
            n_bins_present=hl.len(
                hl.agg.explode(
                    lambda b: hl.agg.collect_as_set(b), pooled_ht.freq_pooled.keys()
                )
            ),
            n_keys_not_released=hl.agg.sum(
                hl.len(entries.filter(lambda kv: ~released_lit.contains(kv[0])))
            ),
            n_an_zero=hl.agg.sum(hl.len(entries.filter(lambda kv: kv[1].AN == 0))),
            n_ac_gt_an=hl.agg.sum(
                hl.len(entries.filter(lambda kv: kv[1].AC > kv[1].AN))
            ),
        )
    )
    results["check4_pooled_release"] = _to_py(agg)
    results["check4_pooled_release"]["n_bins_released"] = len(released)
    results["check4_pooled_release"]["min_released_bin_count"] = min(
        (pooled_counts[b] for b in released), default=None
    )
    return results


def _to_py(value):
    """Convert Hail Structs returned by aggregate into plain Python objects."""
    if isinstance(value, hl.Struct):
        return {k: _to_py(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_py(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted(_to_py(v) for v in value)
    if isinstance(value, dict):
        return {k: _to_py(v) for k, v in value.items()}
    return value


def export_train_pop_tsv(pop_ht: hl.Table, pops: List[str], path: str) -> None:
    """
    Export Experiment 1: training-fold AC/AN per group, one row per variant.

    :param pop_ht: Output of --compute-pop-freq.
    :param pops: Genetic ancestry groups.
    :param path: Output TSV path.
    """
    groups = pops + [ALL_GROUP]
    fields = {}
    for g in groups:
        fields[f"AC_{g}"] = pop_ht.freq_train[g].AC
        fields[f"AN_{g}"] = pop_ht.freq_train[g].AN
    pop_ht.select(**fields).export(path)


def export_pooled_bin_tsv(pooled_ht: hl.Table, path: str) -> None:
    """
    Export Experiment 2: pooled per-bin AC/AN, one row per variant and released bin.

    :param pooled_ht: Output of --compute-pooled-bin-freq.
    :param path: Output TSV path.
    """
    ht = pooled_ht.select(entry=pooled_ht.freq_pooled.items())
    ht = ht.explode("entry")
    ht = ht.select(bin_id=ht.entry[0], AC=ht.entry[1].AC, AN=ht.entry[1].AN)
    ht.export(path)


def main(args):
    """Run the GridMax companion frequency pipeline."""
    hl.init(log="/tmp/gridmax_pop_freq.log", tmp_dir=args.tmp_dir)
    chrom = args.chrom
    test_gene = args.test_gene
    overwrite = args.overwrite
    paths = get_output_paths(args.output_dir, args.tmp_dir, chrom, test_gene)
    intervals = get_scope_intervals(chrom, test_gene)

    steps = [
        "prep_vds",
        "densify",
        "compute_bin_freq",
        "compute_pop_freq",
        "compute_pooled_bin_freq",
        "validate",
        "export",
    ]
    run_all = not any(getattr(args, s) for s in steps)

    def run(step):
        return run_all or getattr(args, step)

    training_ht = get_gridmax_training_samples().ht()
    binned_exomes = training_ht.filter(
        hl.is_defined(training_ht.bin_id) & (training_ht.data_type == "exome")
    )
    pops = sorted(
        p
        for p in binned_exomes.aggregate(hl.agg.collect_as_set(binned_exomes.pop))
        if p is not None
    )
    logger.info("Genetic ancestry groups: %s", pops)

    if args.use_production_bin_freq:
        bin_freq_path = get_gridmax_freq(
            step="freq", chrom=chrom, test_gene=test_gene
        ).path
    else:
        bin_freq_path = paths["bin_freq"]
    logger.info("Per-fold per-bin call stats: %s", bin_freq_path)

    if run("prep_vds"):
        logger.info("Step: prep and split VDS...")
        vds = prep_vds_for_gridmax(
            training_ht, chrom=chrom, test_gene=test_gene, keep_pop=True
        )
        vds.checkpoint(paths["split_vds"], overwrite=overwrite)

    if run("densify"):
        logger.info("Step: densify split VDS...")
        mt = densify_split_vds(hl.vds.read_vds(paths["split_vds"]), args.ab_cutoff)
        mt.checkpoint(paths["dense_mt"], overwrite=overwrite)

    if run("compute_bin_freq") and not args.use_production_bin_freq:
        logger.info("Step: per-bin per-fold call stats...")
        ht = aggregate_pixel_freq(hl.read_matrix_table(paths["dense_mt"]))
        ht.checkpoint(paths["bin_freq"], overwrite=overwrite)

    if run("compute_pop_freq"):
        logger.info("Step: per-pop call stats...")
        ht = compute_pop_freq(
            hl.read_matrix_table(paths["dense_mt"]),
            pops,
            af_threshold=args.af_threshold,
            pooled_bins_direct=args.pooled_bins_direct,
        )
        ht.checkpoint(paths["pop_freq"], overwrite=overwrite)

    pooled_counts = None
    if run("compute_pooled_bin_freq") or run("validate"):
        pooled_counts = get_pooled_bin_counts(training_ht)

    if run("compute_pooled_bin_freq"):
        logger.info("Step: pooled per-bin call stats...")
        ht = compute_pooled_bin_freq(
            hl.read_table(bin_freq_path),
            pooled_counts,
            af_threshold=args.af_threshold,
            min_bin_occupancy=args.min_bin_occupancy,
        )
        ht.checkpoint(paths["pooled_bin_freq"], overwrite=overwrite)

    if run("validate"):
        logger.info("Step: validation...")
        results = validate(
            hl.read_table(paths["pop_freq"]),
            hl.read_table(bin_freq_path),
            hl.read_table(paths["pooled_bin_freq"]),
            pooled_counts,
            pops,
            args.af_threshold,
            args.min_bin_occupancy,
            intervals=intervals,
        )
        with hl.hadoop_open(paths["validation"], "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info("Validation written to %s", paths["validation"])

    if run("export"):
        logger.info("Step: export TSVs...")
        export_train_pop_tsv(
            hl.read_table(paths["pop_freq"]), pops, paths["train_pop_tsv"]
        )
        export_pooled_bin_tsv(
            hl.read_table(paths["pooled_bin_freq"]), paths["pooled_bin_tsv"]
        )
        logger.info("Exports written to %s", args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    step_group = parser.add_argument_group(
        "pipeline steps",
        "Select one or more steps to run. If none are specified, all steps run in order.",
    )
    step_group.add_argument(
        "--prep-vds", action="store_true", help="Prep and split VDS."
    )
    step_group.add_argument("--densify", action="store_true", help="Densify split VDS.")
    step_group.add_argument(
        "--compute-bin-freq",
        action="store_true",
        help="Per-bin per-fold call stats (skipped with --use-production-bin-freq).",
    )
    step_group.add_argument(
        "--compute-pop-freq", action="store_true", help="Per-pop call stats."
    )
    step_group.add_argument(
        "--compute-pooled-bin-freq",
        action="store_true",
        help="Pooled per-bin call stats with privacy filter.",
    )
    step_group.add_argument("--validate", action="store_true", help="Run validation.")
    step_group.add_argument("--export", action="store_true", help="Export TSVs.")

    parser.add_argument(
        "--output-dir", required=True, help="Root output directory (GCS path)."
    )
    parser.add_argument(
        "--tmp-dir",
        default="gs://gnomad-tmp-4day",
        help="Hail temporary directory (default: gs://gnomad-tmp-4day).",
    )
    parser.add_argument("--chrom", help="Restrict to this chromosome (e.g. chr20).")
    parser.add_argument(
        "--test-gene",
        action="store_true",
        help=f"Restrict to DRD2 ({TEST_GENE_INTERVAL}).",
    )
    parser.add_argument(
        "--use-production-bin-freq",
        action="store_true",
        help=(
            "Read per-fold per-bin call stats from the production "
            "compute_gridmax_freq.py --compute-freq output instead of computing them."
        ),
    )
    parser.add_argument(
        "--pooled-bins-direct",
        action="store_true",
        help="In --compute-pop-freq, also compute pooled per-bin call stats directly (validation).",
    )
    parser.add_argument(
        "--ab-cutoff",
        type=float,
        default=DEFAULT_AB_CUTOFF,
        help=f"Allele balance cutoff for high-AB hets (default: {DEFAULT_AB_CUTOFF}).",
    )
    parser.add_argument(
        "--af-threshold",
        type=float,
        default=DEFAULT_AF_THRESHOLD,
        help=f"AF threshold for the high-AB-het correction (default: {DEFAULT_AF_THRESHOLD}).",
    )
    parser.add_argument(
        "--min-bin-occupancy",
        type=int,
        default=DEFAULT_MIN_BIN_OCCUPANCY,
        help=f"Privacy threshold for pooled bins (default: {DEFAULT_MIN_BIN_OCCUPANCY}).",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite outputs.")

    main(parser.parse_args())
