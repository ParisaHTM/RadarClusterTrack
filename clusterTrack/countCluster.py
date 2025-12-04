import os
import sys
import glob
import pickle
from typing import Dict, List, Any, Tuple

# Allow running as a script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)

try:
	from config import DEFAULT_PATHS  # type: ignore
except Exception:
	DEFAULT_PATHS = {'pickle_save_root': './clustering_data_pkl_rcs'}


def count_clusters_in_entry(entry: Dict[str, Any]) -> int:
	"""
	Safe cluster count for a single camera-frame entry.
	Prefers 'num_clusters' if present; falls back to len('clusters').
	"""
	if isinstance(entry, dict):
		if 'num_clusters' in entry and isinstance(entry['num_clusters'], (int, float)):
			try:
				return int(entry['num_clusters'])
			except Exception:
				pass
		if 'clusters' in entry and isinstance(entry['clusters'], list):
			return len(entry['clusters'])
	return 0


def count_clusters_in_pickle(pkl_path: str) -> Tuple[int, int]:
	"""
	Load a pickle file and sum cluster counts across its entries.
	Returns (num_entries, total_cluster_instances).
	"""
	with open(pkl_path, 'rb') as f:
		data = pickle.load(f)
	if not isinstance(data, list):
		return 0, 0
	total = 0
	for entry in data:
		total += count_clusters_in_entry(entry)
	return len(data), total


def main(pickle_root: str = None) -> None:
	pickle_root = pickle_root or DEFAULT_PATHS.get('pickle_save_root', './clustering_data_pkl_rcs')
	pickle_root = os.path.abspath(pickle_root)
	if not os.path.isdir(pickle_root):
		print(f"Pickle directory not found: {pickle_root}")
		return

	pkl_files = sorted(glob.glob(os.path.join(pickle_root, "*.pkl")))
	if not pkl_files:
		print(f"No .pkl files found under: {pickle_root}")
		return

	grand_total_clusters = 0
	grand_total_entries = 0

	print(f"Scanning {len(pkl_files)} pickle files in: {pickle_root}")
	for p in pkl_files:
		num_entries, num_clusters = count_clusters_in_pickle(p)
		grand_total_clusters += num_clusters
		grand_total_entries += num_entries
		print(f"- {os.path.basename(p)}: entries={num_entries}, cluster_instances={num_clusters}")

	print("\n=== Summary ===")
	print(f"Files processed: {len(pkl_files)}")
	print(f"Total entries (camera-frames): {grand_total_entries}")
	print(f"Total cluster instances (sum over entries): {grand_total_clusters}")
	if grand_total_entries > 0:
		avg_per_entry = grand_total_clusters / float(grand_total_entries)
		print(f"Average clusters per entry: {avg_per_entry:.3f}")


if __name__ == "__main__":
	# Optional: pass a custom directory as the first CLI arg
	custom_dir = sys.argv[1] if len(sys.argv) > 1 else None
	main(custom_dir)


