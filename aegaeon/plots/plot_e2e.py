import argparse
import os 
import json
from matplotlib import pyplot as plt

def plot_by_models(
    spec: str,
    rate: str,
    inlen_scaled: bool = False,
    outlen_scaled: bool = False,
    ttft_slo: float = 10.0,
    tpot_slo: float = 0.1,
):
    qos = {}

    # Collect aegaeon results
    models = range(16, 81, 4) if rate == "0.1" else range(16, 55, 2)
    json_files = [
        f"json/run-{m}-{rate}-{2.0 if inlen_scaled else 1.0}-{2.0 if outlen_scaled else 1.0}-ttft{ttft_slo}-tpot{tpot_slo}.json"
        for m in models
    ]
    slo_aegaeon = []
    for m, json_file in zip(models, json_files):
        if not os.path.exists(json_file):
            print(f'[INFO] Result file {json_file} does not exist; skipping this sample')
            continue
        slo_aegaeon.append((m, compute_slo(json_file, ttft_slo, tpot_slo)))
    qos['Aegaeon'] = {
        float(m): slo for m, slo in slo_aegaeon
    }

    # Collect sllm/sllm+/muxserve results
    qos |= compute_slo_sim(
        f"json/sim-model{models[0]}-{models[-1]}-rate{rate}-scale-{2.0 if inlen_scaled else 1.0}-{2.0 if outlen_scaled else 1.0}.json",
        ttft_slo,
        tpot_slo,
    )
    qos['MuxServe'] = {
        m: 1 for m in models if m <= 32
    }

    if spec.startswith("slo") and "SLLM+" in qos:
        qos.pop("SLLM+")

    print(qos)

    # Plot
    plt.rcParams.update({'font.size': 18})
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.set_xlabel("#Models")
    ax.set_ylabel("SLO Attainment (%)")

    if rate == "0.1":
        ax.set_xlim(10, 85)
        ax.set_xticks([20, 40, 60, 80])
        ax.set_xticklabels(ax.get_xticks())
        ax.set_ylim(0, 105)
        ax.set_yticks([0, 50, 100])
        ax.set_yticklabels(ax.get_yticks())
    else:
        ax.set_xlim(15, 56)
        ax.set_xticks([16, 24, 32, 40, 48 ])
        ax.set_xticklabels(ax.get_xticks())
        ax.set_ylim(0, 105)
        ax.set_yticks([0, 50, 100])
        ax.set_yticklabels(ax.get_yticks())

    ax.axhline(y=90, color="tab:gray", linestyle="--", linewidth=1)

    for policy, data in qos.items():
        X = [x for x, _ in data.items()]
        yvals = [y * 100 for _, y in data.items()]
        ax.plot(X, yvals, "o-", label=f"{policy}")

    fig.legend()
    plt.tight_layout(pad=0.2)
    plt.savefig(f'{spec}.pdf')


def plot_by_rates():
    qos = {}

    # Collect aegaeon results
    rates = [x*0.05 for x in range(1, 15)]
    json_files = [
        f"json/run-40-{rate}-1.0-1.0-ttft10.0-tpot0.1.json"
        for rate in rates
    ]
    slo_aegaeon = []
    for rate, json_file in zip(rates, json_files):
        if not os.path.exists(json_file):
            print(f'[INFO] Result file {json_file} does not exist; skipping this sample')
            continue
        slo_aegaeon.append((rate, compute_slo(json_file, 10.0, 0.1)))
    qos['Aegaeon'] = {
        rate: slo for rate, slo in slo_aegaeon
    }

    # Collect sllm/sllm+ results
    qos |= compute_slo_sim(
        f"json/sim-model40-rate0.05-0.70-scale-1.0-1.0.json",
        10.0,
        0.1,
    )

    print(qos)

    # Plot
    plt.rcParams.update({'font.size': 18})
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.set_xlabel('Average Arrival Rate (req/s)', fontdict=fontdict)
    ax.set_ylabel("SLO Attainment (%)")
    ax.set_xlim(0, 0.75)
    ax.set_xticks([0, 0.2, 0.4, 0.6])
    ax.set_xticklabels(ax.get_xticks(), fontdict=fontdict)
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 50, 100])
    ax.set_yticklabels(ax.get_yticks(), fontdict=fontdict)

    ax.axhline(y=90, color="tab:gray", linestyle="--", linewidth=1)

    for policy, data in qos.items():
        X = [x for x, _ in data.items()]
        yvals = [y * 100 for _, y in data.items()]
        ax.plot(X, yvals, "o-", label=f"{policy}")

    fig.legend(loc='auto')
    plt.tight_layout(pad=0.2)
    plt.savefig(f'40.pdf')


def compute_slo(
    json_file,
    ttft_slo: float,
    tpot_slo: float,
) -> float:
    
    # Compute SLO Attainment
    TTFT_SLO = ttft_slo
    TPOT_SLO = tpot_slo

    with open(json_file, 'r') as f:
        req_metrics = json.load(f)
   
    # QoS is defined by the pecentage of time where TTFT/TPOT requirements
    # are satisfied over total service time.
    # For example, a request arriving at 0 with each token time (3, 6, 7, 8)
    # and (TTFT_SLO, TPOT_SLO) = (3, 2) has QoS 5/8 from interval (0, 3) and
    # (6, 8).
    
    qos_list = []
    for req_id, metrics in req_metrics.items():

        per_token = metrics["per_token"]
        
        qos = 0
        start = 0
        accum = 0
        target = TTFT_SLO
        sum_token = 0
        for token in per_token:
            sum_token += token
            if sum_token - start <= target:
                accum = target
                target += TPOT_SLO
            else:
                qos += accum
                start = sum_token
                accum = 0
                target += TPOT_SLO
        qos += min(target, sum(per_token)-start)
        if per_token:
            qos_list.append(qos / sum(per_token))

    return sum(qos_list)/len(qos_list)

def compute_slo_sim(
    json_file,
    ttft_slo: float,
    tpot_slo: float,
) -> dict:
    all_per_token_ms = {}
    NMODELS = set()
    POLICIES = set()
    with open(f'{json_file}', 'r') as f:
        metrics = json.load(f)
        for k, v in metrics.items():
            m, _policy = k[1:-1].split(',')
            model = float(m)
            NMODELS.add(model)
            if 'SeLLMPolicy' in _policy:
                policy = "SLLM"
            elif 'SeLLMPlusPolicy' in _policy:
                policy = "SLLM+"
            else:
                raise ValueError(f'unmmaped policy: {_policy}')
            POLICIES.add(policy)
            all_per_token_ms[(model, policy)] = v

    NMODELS = sorted(list(NMODELS))
    POLICIES = sorted(list(POLICIES))

    # Plot SLO Attainment
    TTFT_SLO = 1000 * ttft_slo
    TPOT_SLO = 1000 * tpot_slo
    qos_by_m = {policy: {m: 0 for m in NMODELS} for policy in POLICIES}

    for policy in POLICIES:
        for m in sorted(list(NMODELS)):
            req_per_token_ms = all_per_token_ms[(m, policy)]
            qos_list = []
            for req_id, per_token_ms in req_per_token_ms.items():
                if per_token_ms[0] == -1:
                    qos_list.append(0)
                    continue
                
                qos = 0
                start_ms = 0
                accum = 0
                target = TTFT_SLO
                sum_token_ms = 0
                for token_ms in per_token_ms:
                    sum_token_ms += token_ms
                    if sum_token_ms - start_ms <= target:
                        accum = target
                        target += TPOT_SLO
                    else:
                        qos += accum
                        start_ms = sum_token_ms
                        accum = 0
                        target += TPOT_SLO
                qos += min(target, sum(per_token_ms)-start_ms)
                qos_list.append(qos / sum(per_token_ms))
            print(policy, m)
            qos_by_m[policy][m] = sum(qos_list) / len(qos_list)
    
    return qos_by_m


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', choices=["0.1", "0.5", "40", "0.1-ix2", "0.1-ox2", "0.5-ix2", "0.5-ox2", "slo0.5", "slo0.3"])
    args = parser.parse_args()

    if args.spec == "40":
        plot_by_rates()
    else:
        if args.spec.endswith("-ix2"):
            plot_by_models(
                args.spec, args.spec[:3], inlen_scaled=True, ttft_slo=10.0, tpot_slo=0.1,
            )
        elif args.spec.endswith("-ox2"):
            plot_by_models(
                args.spec, args.spec[:3], outlen_scaled=True, ttft_slo=10.0, tpot_slo=0.1,
            )
        elif args.spec.startswith("slo"):
            scale = float(args.spec[3:])
            plot_by_models(
                args.spec, "0.1", ttft_slo=10.0*scale, tpot_slo=0.1*scale,
            )
        else:
            plot_by_models(
                args.spec, args.spec, ttft_slo=10.0, tpot_slo=0.1,
            )
