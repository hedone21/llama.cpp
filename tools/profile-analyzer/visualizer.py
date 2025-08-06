#!/usr/bin/env python3

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np


# Define the data structure for a single JSON object.
# Using dataclass automatically creates the constructor (__init__).
@dataclass
class Operation:
    """Operation data structure"""

    name: str
    type: str
    op: str
    shape: list[int]
    start_at: int
    duration: int
    memory: int
    start_core: int
    end_core: int
    core_changed: bool


def parse_operations_from_json(json_string: str) -> list[Operation]:
    """
    Parses a JSON string into a list of Operation objects.

    Args:
        json_string: The JSON-formatted string to parse.

    Returns:
        A list of Operation objects containing the parsed data.
    """
    try:
        # Use json.loads() to convert the JSON string into a Python object (a list of dictionaries).
        data = json.loads(json_string)

        # Use list comprehension and dictionary unpacking (**) to convert each dictionary into an Operation object.
        operations_list = [Operation(**item) for item in data]

        prev_name_list = []
        prev_name = None
        for i, op in enumerate(operations_list):
            if prev_name != op.name:
                if op.name in prev_name_list:
                    op.name = f"{op.name}-duplicated"

            prev_name = op.name
            prev_name_list.append(op.name)

        return operations_list
    except json.JSONDecodeError:
        print("Error: Invalid JSON format.")
        return []
    except TypeError as e:
        print(f"Error: JSON data keys do not match the Operation structure. ({e})")
        return []


def calculate_busy_time(ops_group: list[Operation]) -> int:
    """
    Calculates the total busy time by merging overlapping intervals.
    This represents the actual time that at least one thread was working.
    """
    if not ops_group:
        return 0

    # Create a list of intervals [start, end] and sort by start time.
    intervals = [(op.duration) for op in ops_group]

    return sum(intervals)


def visualize_operations_as_gantt(
    operations: list[Operation], start: int = 0, n: int = -1
):
    """
    Visualizes the parsed list of Operations as a Gantt chart.
    Tasks with the same name are displayed in the same row to show parallelism.
    """
    if not operations:
        print("No data to visualize.")
        return

    # 1. Group operations by their 'name'.
    grouped_ops: dict[str, list[Operation]] = defaultdict(list)
    prev_name = None
    i = 0
    for op in operations:
        # If n is specified, limit the number of operations processed.
        if prev_name != op.name:
            prev_name = op.name
            i += 1

        if n > 0 and i > n + start:
            continue

        if i < start:
            continue

        grouped_ops[op.name].append(op)

    # 2. Find the earliest start time to normalize the timeline.
    min_start_at = min(op.start_at for op in operations)

    # 3. Calculate bubble time and prepare new y-axis labels
    op_names = list(grouped_ops.keys())
    new_yticklabels = []
    total_mean_busy_time = 0
    start_at = 0
    end_at = 0
    print("--- Bubble Time Analysis ---")
    for name in op_names:
        ops_in_group = grouped_ops[name]

        # Calculate total span for the group
        group_start = min(op.start_at for op in ops_in_group)
        group_end = max(op.start_at + op.duration for op in ops_in_group)
        if start_at == 0 or group_start < start_at:
            start_at = group_start
        if end_at == 0 or group_end > end_at:
            end_at = group_end
        total_span = (group_end - group_start) * len(ops_in_group)
        print(f"Processing group '{name}' with {len(ops_in_group)} operations...")
        print(
            f"  - Group Start: {group_start}, Group End: {group_end}, Total Span: {total_span}"
        )

        # Calculate total busy time by merging intervals
        busy_time = calculate_busy_time(ops_in_group)
        total_mean_busy_time += busy_time / len(ops_in_group)

        # Calculate bubble time and percentage
        bubble_time = total_span - busy_time

        bubble_percentage = (bubble_time / total_span) * 100 if total_span > 0 else 0

        # Find the longest time in the group for display purposes
        longest_time = max(op.duration for op in ops_in_group)

        print(f"'{name}':")
        print(f"  - Total Span: {total_span} units")
        print(f"  - Busy Time (work done): {busy_time} units")
        print(f"  - Bubble Time (idle): {bubble_time} units")
        print(f"  - Bubble Percentage: {bubble_percentage:.2f}%")
        print("-" * 28)

        # Create the new label for the y-axis
        new_yticklabels.append(
            f"{name} ({longest_time}us / {len(ops_in_group)} threads / {bubble_percentage:.1f}% bubble)"
        )
    print("\nChart is being generated...")

    print("--- Summary ---")
    print(f"Total Busy Time: {total_mean_busy_time:.0f} us")
    print(f"Start Time: {start_at} us")
    print(f"End Time: {end_at} us")
    print(f"Total Duration: {end_at - start_at} us")
    print(
        f"Busy Time Percentage: {(total_mean_busy_time / (end_at - start_at)) * 100:.2f}%"
    )

    # 4. Set up the Matplotlib chart.
    fig, ax = plt.subplots(figsize=(15, 8))

    # Set up the y-axis labels (operation names) and their positions.
    op_names = list(grouped_ops.keys())
    y_pos = np.arange(len(op_names))

    # 5. Add bars to the chart for each group (by name).
    for i, name in enumerate(op_names):
        # Use a color map to assign different colors to threads within the same operation group.
        colors = plt.cm.viridis(np.linspace(0, 1, len(grouped_ops[name])))

        ops = grouped_ops[name]
        ops = sorted(
            ops, key=lambda op: op.duration, reverse=True
        )  # Sort by time for better visualization.

        for j, op_instance in enumerate(ops):
            # Calculate the bar's start time and duration (normalized).
            start_time_normalized = op_instance.start_at - min_start_at
            duration = op_instance.duration

            # Draw the horizontal bar (barh).
            # y: The y-coordinate of the bar.
            # width: The length of the bar (task duration).
            # left: The starting x-coordinate of the bar.
            ax.barh(
                y_pos[i],
                width=duration,
                left=start_time_normalized,
                height=0.1,
                align="center",
                color=colors[j],
                edgecolor="black",
                label=f"Thread {j+1}" if i == 0 else "",
            )  # Label threads only for the first group for a cleaner legend.

    # 6. Customize the chart's design and labels.
    ax.set_yticks(y_pos)
    ax.set_yticklabels(new_yticklabels)
    ax.invert_yaxis()

    ax.set_xlabel("Execution Time (Normalized Timestamp)")
    ax.set_title("Parallel Task Execution Timeline (Gantt Chart)", fontsize=16, pad=20)

    # Add a grid for better readability.
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, axis="x")

    # Optional: Add a legend if needed.
    # handles, labels = ax.get_legend_handles_labels()
    # by_label = dict(zip(labels, handles))
    # ax.legend(by_label.values(), by_label.keys())

    plt.tight_layout()
    plt.show()


# --- Example Usage ---
if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Parse JSON operations data")
    args.add_argument(
        "json_file", type=str, help="Path to the JSON file containing operations data"
    )
    args.add_argument(
        "-n",
        type=int,
        help="Number of operations to visualize",
        default=-1,
    )
    args.add_argument(
        "--start",
        type=int,
        help="Start Number of operations to visualize",
        default=0,
    )
    parsed_args = args.parse_args()

    try:
        with open(parsed_args.json_file, "r") as file:
            json_data = file.read()
    except FileNotFoundError:
        print(f"Error: The file {parsed_args.json_file} does not exist.")
        exit(1)

    # 1. Parse the JSON data.
    operations = parse_operations_from_json(json_data)

    # 2. Visualize the parsed data as a Gantt chart.
    if operations:
        visualize_operations_as_gantt(operations, parsed_args.start, parsed_args.n)
