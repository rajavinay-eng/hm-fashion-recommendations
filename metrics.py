def apk(actual, predicted, k=12):
    """Average precision at k for one customer."""
    if not actual:
        return 0.0
    actual = set(actual)
    hits, score = 0, 0.0
    seen = set()
    for i, p in enumerate(predicted[:k]):
        if p in actual and p not in seen:
            hits += 1
            score += hits / (i + 1)
        seen.add(p)
    return score / min(len(actual), k)


def mapk(actual_dict, predicted_dict, k=12):
    """Mean AP@k over customers in actual_dict.
    Customers with no prediction get 0."""
    scores = [apk(actual, predicted_dict.get(cid, []), k)
              for cid, actual in actual_dict.items()]
    return sum(scores) / len(scores)