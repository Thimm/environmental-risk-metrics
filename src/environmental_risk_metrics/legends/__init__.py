def convert_legend_to_value_color_dict(legend: dict) -> dict:
    return {item["value"]: item["color"] for item in legend.values()}


def convert_legend_to_value_label_dict(legend: dict) -> dict:
    return {item["value"]: item["label"] for item in legend.values()}
