-- Motor + agrafana + таблицы Querulus monitoring (как в контуре)
IF DB_ID(N'Motor') IS NULL
BEGIN
    CREATE DATABASE Motor;
END
GO

USE Motor;
GO

IF NOT EXISTS (SELECT 1 FROM sys.schemas WHERE name = N'agrafana')
BEGIN
    EXEC(N'CREATE SCHEMA [agrafana]');
END
GO

IF OBJECT_ID(N'[agrafana].[Querulus_errors]', N'U') IS NULL
BEGIN
    CREATE TABLE [agrafana].[Querulus_errors] (
        [run_date] DATE NULL,
        [stem] NVARCHAR(256) NULL,
        [slot] NVARCHAR(64) NULL,
        [model_label] NVARCHAR(256) NULL,
        [model_kind] NVARCHAR(256) NULL,
        [model_version] NVARCHAR(256) NULL,
        [error_fingerprint] NVARCHAR(64) NULL,
        [error_head] NVARCHAR(MAX) NULL,
        [count] INT NULL
    );
END
GO

IF OBJECT_ID(N'[agrafana].[querulus_DataDrift]', N'U') IS NULL
BEGIN
    CREATE TABLE [agrafana].[querulus_DataDrift] (
        [model_name] NVARCHAR(256) NULL,
        [model_version] NVARCHAR(256) NULL,
        [date] DATE NULL,
        [col] NVARCHAR(256) NULL,
        [TYPE] NVARCHAR(64) NULL,
        [PSI] FLOAT NULL,
        [KL] FLOAT NULL,
        [JS] FLOAT NULL,
        [NaN_train] FLOAT NULL,
        [NaN_test] FLOAT NULL,
        [uniq_train] FLOAT NULL,
        [uniq_test] FLOAT NULL,
        [mode_train] NVARCHAR(MAX) NULL,
        [mode_test] NVARCHAR(MAX) NULL,
        [mean_train] FLOAT NULL,
        [mean_test] FLOAT NULL,
        [dif_train] NVARCHAR(MAX) NULL,
        [dif_test] NVARCHAR(MAX) NULL,
        [stem] NVARCHAR(256) NULL,
        [track] NVARCHAR(64) NULL,
        [model_label] NVARCHAR(256) NULL
    );
END
GO

IF OBJECT_ID(N'[agrafana].[Querulus_FinEffect]', N'U') IS NULL
BEGIN
    CREATE TABLE [agrafana].[Querulus_FinEffect] (
        [week_start] DATE NULL,
        [stem] NVARCHAR(256) NULL,
        [slot] NVARCHAR(64) NULL,
        [model_label] NVARCHAR(256) NULL,
        [filial] NVARCHAR(256) NULL,
        [n_incidents] INT NULL,
        [n_model] INT NULL,
        [n_control] INT NULL,
        [itt] FLOAT NULL,
        [net] FLOAT NULL,
        [coverage] FLOAT NULL,
        [formula_version] NVARCHAR(128) NULL,
        [annual_window_yult] FLOAT NULL,
        [annual_pilot_current_yult] FLOAT NULL,
        [annual_pilot_full_yult] FLOAT NULL,
        [annual_network_current_yult] FLOAT NULL,
        [annual_network_full_yult] FLOAT NULL,
        [annual_network_full_compliance_yult] FLOAT NULL,
        [agreement_share_control] FLOAT NULL,
        [agreement_share_model] FLOAT NULL,
        [agreement_share_lift_pp] FLOAT NULL,
        [pretension_share_control] FLOAT NULL,
        [pretension_share_model] FLOAT NULL,
        [pretension_share_lift_pp] FLOAT NULL,
        [fu_share_control] FLOAT NULL,
        [fu_share_model] FLOAT NULL,
        [fu_share_lift_pp] FLOAT NULL,
        [court_share_control] FLOAT NULL,
        [court_share_model] FLOAT NULL,
        [court_share_lift_pp] FLOAT NULL,
        [agr_control_pct] FLOAT NULL,
        [agr_model_pct] FLOAT NULL,
        [gap_pp] FLOAT NULL
    );
END
GO
