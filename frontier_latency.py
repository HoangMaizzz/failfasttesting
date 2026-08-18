def bucket_floor(value, bucket_size):
    value = max(0, int(value))
    bucket_size = max(1, int(bucket_size))
    return value // bucket_size * bucket_size


def create_verify_latency_model(context_bucket_size=256, proposal_bucket_size=8):
    return {
        "context_bucket_size": max(1, int(context_bucket_size)),
        "proposal_bucket_size": max(1, int(proposal_bucket_size)),
        "joint": {},
        "proposal": {},
        "round_ema_ms": None,
        "observations": 0,
    }


def _update_record(table, key, latency_ms, alpha):
    record = table.get(key)
    if record is None:
        table[key] = [float(latency_ms), 1]
        return
    record[0] = (1.0 - alpha) * float(record[0]) + alpha * float(latency_ms)
    record[1] = int(record[1]) + 1


def update_verify_latency_model(model, context_len, proposal_len, latency_ms, alpha=0.2):
    if model is None:
        model = create_verify_latency_model()
    if latency_ms <= 0 or proposal_len <= 0:
        return model

    alpha = min(1.0, max(0.0, float(alpha)))
    context_bucket = bucket_floor(context_len, model["context_bucket_size"])
    proposal_bucket = bucket_floor(proposal_len, model["proposal_bucket_size"])
    joint_key = f"{context_bucket}:{proposal_bucket}"
    proposal_key = str(proposal_bucket)
    _update_record(model["joint"], joint_key, latency_ms, alpha)
    _update_record(model["proposal"], proposal_key, latency_ms, alpha)

    old_round_ema = model.get("round_ema_ms")
    model["round_ema_ms"] = (
        float(latency_ms)
        if old_round_ema is None
        else (1.0 - alpha) * float(old_round_ema) + alpha * float(latency_ms)
    )
    model["observations"] = int(model.get("observations", 0)) + 1
    return model


def _interpolate(points, target):
    points = sorted(points, key=lambda item: item[0])
    if not points:
        return None
    if len(points) == 1:
        return points[0][1], points[0][2]

    if target <= points[0][0]:
        return points[0][1], points[0][2]
    if target >= points[-1][0]:
        left, right = points[-2], points[-1]
        slope = max(0.0, (right[1] - left[1]) / max(1, right[0] - left[0]))
        return right[1] + slope * (target - right[0]), right[2]

    for left, right in zip(points, points[1:]):
        if left[0] <= target <= right[0]:
            fraction = (target - left[0]) / max(1, right[0] - left[0])
            latency = left[1] + fraction * (right[1] - left[1])
            return latency, min(left[2], right[2])
    return points[-1][1], points[-1][2]


def estimate_verify_latency_ms(model, context_len, proposal_len, fallback_ms):
    fallback_ms = max(float(fallback_ms), 1e-6)
    if not model or not model.get("joint"):
        return {"latency_ms": fallback_ms, "source": "fallback", "samples": 0}

    context_bucket = bucket_floor(context_len, model["context_bucket_size"])
    proposal_bucket = bucket_floor(proposal_len, model["proposal_bucket_size"])
    exact_key = f"{context_bucket}:{proposal_bucket}"
    exact = model["joint"].get(exact_key)
    if exact is not None:
        return {"latency_ms": max(float(exact[0]), 1e-6), "source": "joint_exact", "samples": int(exact[1])}

    same_context = []
    same_proposal = []
    nearest = None
    nearest_distance = None
    for key, record in model["joint"].items():
        stored_context, stored_proposal = (int(value) for value in key.split(":"))
        point = (stored_proposal, float(record[0]), int(record[1]))
        if stored_context == context_bucket:
            same_context.append(point)
        if stored_proposal == proposal_bucket:
            same_proposal.append((stored_context, float(record[0]), int(record[1])))
        distance = (
            abs(stored_context - context_bucket) / model["context_bucket_size"]
            + abs(stored_proposal - proposal_bucket) / model["proposal_bucket_size"]
        )
        if nearest_distance is None or distance < nearest_distance:
            nearest_distance = distance
            nearest = record

    estimate = _interpolate(same_context, proposal_bucket) if len(same_context) >= 2 else None
    if estimate is not None:
        return {"latency_ms": max(float(estimate[0]), 1e-6), "source": "context_interpolation", "samples": int(estimate[1])}

    estimate = _interpolate(same_proposal, context_bucket)
    if estimate is not None:
        return {"latency_ms": max(float(estimate[0]), 1e-6), "source": "proposal_interpolation", "samples": int(estimate[1])}

    proposal_points = [
        (int(key), float(record[0]), int(record[1]))
        for key, record in model.get("proposal", {}).items()
    ]
    estimate = _interpolate(proposal_points, proposal_bucket)
    if estimate is not None:
        return {"latency_ms": max(float(estimate[0]), 1e-6), "source": "global_proposal", "samples": int(estimate[1])}

    if nearest is not None:
        return {"latency_ms": max(float(nearest[0]), 1e-6), "source": "nearest", "samples": int(nearest[1])}

    round_ema = model.get("round_ema_ms")
    return {
        "latency_ms": max(float(round_ema) if round_ema is not None else fallback_ms, 1e-6),
        "source": "round_ema" if round_ema is not None else "fallback",
        "samples": int(model.get("observations", 0)),
    }
