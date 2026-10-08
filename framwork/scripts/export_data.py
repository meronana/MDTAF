"""Collect numbers from results/ and benchmark/results/ into a single JSON for the site.

If the experiments are re-run and the CSVs change, just re-run this script:
    python framwork/scripts/export_data.py
"""
import csv
import json
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RES = os.path.join(ROOT, 'results')
BENCH = os.path.join(ROOT, 'benchmark', 'results')
OUT = os.path.join(ROOT, 'framwork', 'public', 'data', 'site.json')

# Each result file labels endpoints differently, so the site uses one convention
EP = {'fu': 'fu', 'f_u': 'fu', 'clearance': 'CL', 'CL': 'CL', 'half_life': 't12', 't½': 't12'}


def read_csv(*parts):
    with open(os.path.join(*parts), encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def read_json(*parts):
    with open(os.path.join(*parts), encoding='utf-8') as f:
        return json.load(f)


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def main():
    # Data size (Phase 1)
    datasets = {}
    for r in read_csv(BENCH, 'phase_01_summary.csv'):
        datasets[EP[r['Property']]] = {
            'source': int(r['Source (final)']),
            'target': int(r['Target (final)']),
            'overlap': int(r['Overlap']),
        }

    # Diagnostics (Phase 2-5): input-distribution gap vs label correspondence
    p2 = read_json(BENCH, 'phase_02_inter_similarity.json')
    p3 = read_json(BENCH, 'phase_03_representation_gap.json')
    p4 = read_json(BENCH, 'phase_04_label_correspondence.json')
    p5 = read_json(BENCH, 'phase_05_domain_discriminability.json')
    diagnostics = {}
    for k in ('fu', 'clearance', 'half_life'):
        diagnostics[EP[k]] = {
            'tanimoto': p2[k]['mean'],
            'mmd': p3[k]['mmd_median_heuristic'],
            'domain_auc': p5[k]['avg_auc'],
            'nn_rho': p4[k]['spearman_rho'],
            'nn_rho_ci': p4[k]['rho_ci'],
            'nn_pairs': p4[k]['n_pairs'],
            'within_rho': p4[k]['within_target_rho'],
            'above_null': p4[k]['above_null'],
        }

    # Existing DA benchmark (vs the target-only baseline)
    da = [{'endpoint': EP[r['endpoint']], 'method': r['method'], 'lambda': r['λ'],
           'r2': num(r['R²']), 'rho': num(r['ρ']), 'mae': num(r['MAE']), 'rmse': num(r['RMSE'])}
          for r in read_csv(RES, 'paper_tables', 'table4_da_with_baseline.csv')]

    # Proposed methods (multi-seed, bootstrap CI, Holm correction)
    proposed = [{'endpoint': EP[r['endpoint']], 'method': r['method'],
                 'r2': num(r['R²']), 'rho': num(r['ρ']), 'dr2': num(r['ΔR²']),
                 'ci': [num(r['CI_lo']), num(r['CI_hi'])], 'p': num(r['p']), 'p_holm': num(r['p_holm'])}
                for r in read_csv(RES, 'proposed', 'summary_table_final.csv')]

    # Generalisation to other domain pairs (cross-domain)
    cross = [{'pair': r['pair'], 'nn_rho': num(r['nn_rho']), 'n_source': int(r['n_source']),
              'n_target': int(r['n_target_fit']), 'r2_target_only': num(r['TargetOnly R²']),
              'd_ptft': num(r['ΔPTFT']), 'p_ptft': num(r['p(PTFT)']),
              'd_ptft_shuffled': num(r['ΔPTFT-shuffled']), 'p_ptft_shuffled': num(r['p(PTFT-shuffled)']),
              'd_lcw': num(r['ΔLCW']), 'p_lcw': num(r['p(LCW)']),
              'd_ivp': num(r['ΔIVP']), 'p_ivp': num(r['p(IVP)'])}
             for r in read_csv(RES, 'cross_domain', 'summary_table.csv')]

    external = [{'method': r['method'], 'r2': num(r['R²']), 'rho': num(r['ρ']), 'dr2': num(r['ΔR²']),
                 'ci': [num(r['CI_lo']), num(r['CI_hi'])], 'p': num(r['p']), 'p_holm': num(r['p_holm'])}
                for r in read_csv(RES, 'external_validation', 'summary.csv')]

    site = {'datasets': datasets, 'diagnostics': diagnostics, 'da': da,
            'proposed': proposed, 'cross_domain': cross, 'external': external}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(site, f, ensure_ascii=False, indent=1)
    print('wrote', OUT)


if __name__ == '__main__':
    main()
