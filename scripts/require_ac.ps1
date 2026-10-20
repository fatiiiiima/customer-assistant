# Dot-source at the top of benchmark scripts: refuses to run on battery (results would be throttled).
$b = Get-CimInstance Win32_Battery -ErrorAction SilentlyContinue
if ($b -and $b.BatteryStatus -ne 2 -and $b.BatteryStatus -ne 6 -and $b.BatteryStatus -ne 7 -and $b.BatteryStatus -ne 8) {
    throw "Laptop is on battery (BatteryStatus=$($b.BatteryStatus)). Plug in AC and choose Best performance, then rerun."
}
Write-Host "AC power OK (BatteryStatus=$($b.BatteryStatus))"
