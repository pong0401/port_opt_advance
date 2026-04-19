param(
    [Parameter(Mandatory = $true)]
    [string]$WorkbookPath
)

$ErrorActionPreference = 'Stop'

$connString = "Provider=Microsoft.ACE.OLEDB.12.0;Data Source=$WorkbookPath;Extended Properties='Excel 8.0;HDR=No;IMEX=1'"
$conn = New-Object System.Data.OleDb.OleDbConnection($connString)

try {
    $conn.Open()
    $schema = $conn.GetOleDbSchemaTable([System.Data.OleDb.OleDbSchemaGuid]::Tables, $null)
    $sheet = $schema |
        Select-Object -ExpandProperty TABLE_NAME |
        ForEach-Object { $_.Trim("'") } |
        Where-Object { $_ -like '*SET100*' -and $_ -notlike '*Print_*' -and $_ -notlike '*FilterDatabase*' -and $_ -like '*$' } |
        Select-Object -First 1

    if (-not $sheet) {
        throw "No SET100 worksheet found in $WorkbookPath"
    }

    $cmd = $conn.CreateCommand()
    $cmd.CommandText = "SELECT * FROM [$sheet]"

    $adapter = New-Object System.Data.OleDb.OleDbDataAdapter($cmd)
    $dataset = New-Object System.Data.DataSet
    [void]$adapter.Fill($dataset)
    $table = $dataset.Tables[0]

    $rows = foreach ($row in $table.Rows) {
        $obj = [ordered]@{}
        foreach ($col in $table.Columns) {
            $obj[$col.ColumnName] = [string]$row[$col.ColumnName]
        }
        [pscustomobject]$obj
    }

    $rows | ConvertTo-Json -Compress
}
finally {
    if ($conn.State -eq [System.Data.ConnectionState]::Open) {
        $conn.Close()
    }
}
