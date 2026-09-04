select 
    pr.PretensionNumber PRETENSION_NUMBER
    ,L.LossNumber LOSS_NUMBER
    ,(L.PolicyholderPersonID) as POLICYHOLDER_PERSON_ID
    ,(L.VictimPersonID) as VICTIM_PERSON_ID
    ,(L.VIctimPolicyholderPersonID) as VICTIM_POLICYHOLDER_PERSON_ID
    , L.VictimObjectOwnerPersonID as VICTIM_OBJECT_OWNER_PERSON_ID
 from OISUU_REPORT.DBO.oisuu81_t_Pretensions pr
 left join OISUU_REPORT.DBO.oisuu81_t_Losses L on l.LossID=pr.LossID
