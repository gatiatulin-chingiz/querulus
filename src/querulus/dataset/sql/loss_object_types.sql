WITH po AS (
    SELECT
        po.LossNumber,
        po.RefundFormByApplication,
        ROW_NUMBER() OVER (
            PARTITION BY po.LossNumber
            ORDER BY po.PODateTime ASC, po.ID ASC
        ) AS rn
    FROM [OISUU_report].[dbo].[oisuu81_t_PaymentOrders] AS po
)
SELECT
    l.LossNumber AS LOSS_NUMBER,
    l.VictimObjectType AS VICTIM_OBJECT_TYPE,
    po.RefundFormByApplication AS REFUND_FORM_BY_PAYMENT_ORDER
FROM [OISUU_report].[dbo].[oisuu81_t_Losses] AS l
LEFT JOIN po
    ON po.LossNumber = l.LossNumber
   AND po.rn = 1
WHERE l.InsuranceTypeGroup = '{insurance_type_group}'
  AND l.LossProcess IN ({processes})
  AND l.Risk = '{risk}'
