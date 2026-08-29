def retention_loss(survival, valid_links):
    if survival is None:
        return survival
    weights = valid_links[:, None].expand_as(survival)
    return (survival * weights).sum() / weights.sum().clamp_min(1)
