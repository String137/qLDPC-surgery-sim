"""
timeslice 시각화 - tick별 SVG 저장

사용법:
    python viz_timeslice.py <circuit.stim> [--tick N] [--out-dir DIR]

    --tick N    : 특정 tick만 저장
    --out-dir   : 저장 폴더 (기본: <stem>_timeslices/)
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stim

parser = argparse.ArgumentParser()
parser.add_argument("stim_file")
parser.add_argument("--tick", type=int, default=None)
parser.add_argument("--out-dir", type=str, default=None)
args = parser.parse_args()

circuit = stim.Circuit.from_file(args.stim_file)
stem = os.path.splitext(os.path.basename(args.stim_file))[0]
out_dir = args.out_dir or f"{stem}_timeslices"
os.makedirs(out_dir, exist_ok=True)

ticks = [args.tick] if args.tick is not None else range(circuit.num_ticks)

for t in ticks:
    try:
        svg = circuit.diagram("timeslice-svg", tick=t)
        path = os.path.join(out_dir, f"tick_{t:04d}.svg")
        with open(path, "w") as f:
            f.write(str(svg))
    except Exception as e:
        pass

print(f"saved {len(list(ticks))} SVGs to {out_dir}/")