-- Motor + agrafana (как в контуре)
IF DB_ID(N'Motor') IS NULL
BEGIN
    CREATE DATABASE Motor;
END
GO

USE Motor;
GO

IF NOT EXISTS (SELECT 1 FROM sys.schemas WHERE name = N'agrafana')
BEGIN
    EXEC(N'CREATE SCHEMA agrafana');
END
GO
