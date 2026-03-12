import os


COMPACT_FRAME_DATASETS = {
    "KITTI",
    "Oxford",
    "MIT",
    "WOD",
    "KAIST",
    "KAIST_hetero",
    "TIERS",
    "TIERS_hetero",
}


def _to_scalar(value, default=""):
    if isinstance(value, (list, tuple)):
        return value[0] if value else default
    return value if value is not None else default


def _compact_frame_id(frame_name):
    parts = frame_name.rsplit("_", 1)
    frame = parts[1] if len(parts) == 2 else frame_name
    return f"{int(frame):06d}" if str(frame).isdigit() else frame


def _clean_name(path_like):
    if not path_like:
        return "-"
    return os.path.splitext(os.path.basename(path_like))[0]


def resolve_sample_display_fields(data_source, fallback_dataset_name=""):
    """
    Build consistent display fields for progress and failure logging.
    """
    src_raw = _to_scalar(data_source.get("src_id", ""), "")
    tgt_raw = _to_scalar(data_source.get("tgt_id", ""), "")
    scene_raw = _to_scalar(data_source.get("scene_name", "-"), "-")
    sensor_raw = _to_scalar(data_source.get("sensor", ""), "")
    dataset_raw = _to_scalar(data_source.get("dataset_names", fallback_dataset_name), fallback_dataset_name)

    src_str = str(src_raw) if src_raw is not None else ""
    tgt_str = str(tgt_raw) if tgt_raw is not None else ""
    scene_name = str(scene_raw) if scene_raw is not None and str(scene_raw) != "" else "-"
    sensor_name = str(sensor_raw) if sensor_raw is not None and str(sensor_raw) != "" else ""
    dataset_name = str(dataset_raw) if dataset_raw is not None else str(fallback_dataset_name)

    src_name = _clean_name(src_str)
    tgt_name = _clean_name(tgt_str)

    if dataset_name in COMPACT_FRAME_DATASETS:
        src_disp = _compact_frame_id(src_name)
        tgt_disp = _compact_frame_id(tgt_name)
        fail_src = src_disp
        fail_tgt = tgt_disp
    else:
        src_disp = src_name
        tgt_disp = tgt_name
        fail_src = src_str
        fail_tgt = tgt_str

    return {
        "scene_name": scene_name,
        "sensor_name": sensor_name,
        "src_disp": src_disp,
        "tgt_disp": tgt_disp,
        "fail_src": fail_src,
        "fail_tgt": fail_tgt,
    }
