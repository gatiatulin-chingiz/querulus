({icnl_alias}.ClaimOrigin in ({origins}) or {icnl_alias}.ClaimOrigin is null)
      AND ({icnl_alias}.ClaimItem not in ({excluded}) or {icnl_alias}.ClaimItem is null)
      AND {loss_alias}.LossProcess IN ({processes})
