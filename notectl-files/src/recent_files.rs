use std::path::Path;
use std::time::SystemTime;

/// Parse an ISO 8601 datetime string to seconds since Unix epoch (UTC).
/// Handles: YYYY-MM-DDTHH:MM:SS±HH:MM and YYYY-MM-DDTHH:MM:SSZ
pub fn parse_iso8601_to_unix(s: &str) -> Option<i64> {
    if s.len() < 19 || s.as_bytes().get(10) != Some(&b'T') {
        return None;
    }

    let year: i64 = s[0..4].parse().ok()?;
    let month: i64 = s[5..7].parse().ok()?;
    let day: i64 = s[8..10].parse().ok()?;
    let hour: i64 = s[11..13].parse().ok()?;
    let minute: i64 = s[14..16].parse().ok()?;
    let second: i64 = s[17..19].parse().ok()?;

    let tz_offset_secs: i64 = match s.get(19..) {
        None | Some("") | Some("Z") => 0,
        Some(tz) if tz.len() >= 6 => {
            let sign: i64 = if tz.starts_with('-') { -1 } else { 1 };
            let tz_hour: i64 = tz[1..3].parse().ok()?;
            let tz_min: i64 = tz[4..6].parse().ok()?;
            sign * (tz_hour * 3600 + tz_min * 60)
        }
        _ => return None,
    };

    let days = days_from_epoch(year, month, day);
    let local_ts = days * 86400 + hour * 3600 + minute * 60 + second;
    Some(local_ts - tz_offset_secs)
}

/// Extract the `updated:` field value from YAML frontmatter by scanning lines directly.
/// Returns the raw string value so the original timezone is preserved in output.
pub fn extract_updated_from_frontmatter(content: &str) -> Option<String> {
    let mut lines = content.lines();

    if lines.next()?.trim() != "---" {
        return None;
    }

    for line in lines {
        if line.trim() == "---" {
            break;
        }
        if let Some(rest) = line.strip_prefix("updated:") {
            let value = rest.trim().trim_matches('"').trim_matches('\'');
            if !value.is_empty() {
                return Some(value.to_string());
            }
        }
    }

    None
}

/// Convert a SystemTime to seconds since Unix epoch.
pub fn system_time_to_unix(t: SystemTime) -> Option<i64> {
    t.duration_since(SystemTime::UNIX_EPOCH)
        .ok()
        .map(|d| d.as_secs() as i64)
}

/// Format a unix timestamp as a UTC ISO 8601 string (YYYY-MM-DDTHH:MM:SSZ).
pub fn unix_to_iso8601(ts: i64) -> String {
    let time_secs = ts.rem_euclid(86400);
    let days = (ts - time_secs) / 86400;

    let hour = time_secs / 3600;
    let minute = (time_secs % 3600) / 60;
    let second = time_secs % 60;

    let (year, month, day) = days_to_ymd(days);
    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
        year, month, day, hour, minute, second
    )
}

/// Get the best-available unix timestamp for a file.
/// Tries frontmatter `updated:` first, falls back to filesystem mtime.
/// Returns `(unix_ts, "frontmatter" | "mtime", display_string)`.
pub fn file_timestamp(path: &Path) -> Option<(i64, &'static str, String)> {
    if let Ok(content) = std::fs::read_to_string(path)
        && let Some(updated) = extract_updated_from_frontmatter(&content)
        && let Some(ts) = parse_iso8601_to_unix(&updated)
    {
        return Some((ts, "frontmatter", updated));
    }

    let mtime = std::fs::metadata(path).ok()?.modified().ok()?;
    let ts = system_time_to_unix(mtime)?;
    Some((ts, "mtime", unix_to_iso8601(ts)))
}

fn is_leap_year(year: i64) -> bool {
    (year % 4 == 0 && year % 100 != 0) || year % 400 == 0
}

fn days_in_month(year: i64, month: i64) -> i64 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 => {
            if is_leap_year(year) {
                29
            } else {
                28
            }
        }
        _ => 0,
    }
}

fn days_from_epoch(year: i64, month: i64, day: i64) -> i64 {
    let mut days: i64 = 0;
    for y in 1970..year {
        days += if is_leap_year(y) { 366 } else { 365 };
    }
    for m in 1..month {
        days += days_in_month(year, m);
    }
    days + day - 1
}

fn days_to_ymd(mut days: i64) -> (i64, i64, i64) {
    let mut year = 1970i64;
    loop {
        let in_year = if is_leap_year(year) { 366 } else { 365 };
        if days < in_year {
            break;
        }
        days -= in_year;
        year += 1;
    }
    let mut month = 1i64;
    loop {
        let in_month = days_in_month(year, month);
        if days < in_month {
            break;
        }
        days -= in_month;
        month += 1;
    }
    (year, month, days + 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_iso8601_with_offset() {
        let ts = parse_iso8601_to_unix("2026-04-24T23:45:37-05:00").unwrap();
        // 2026-04-25T04:45:37Z in UTC
        let expected = parse_iso8601_to_unix("2026-04-25T04:45:37Z").unwrap();
        assert_eq!(ts, expected);
    }

    #[test]
    fn test_parse_iso8601_z() {
        let ts = parse_iso8601_to_unix("2026-01-01T00:00:00Z").unwrap();
        // Days from 1970-01-01 to 2026-01-01
        assert!(ts > 0);
        let back = unix_to_iso8601(ts);
        assert_eq!(back, "2026-01-01T00:00:00Z");
    }

    #[test]
    fn test_parse_iso8601_positive_offset() {
        let ts = parse_iso8601_to_unix("2026-04-25T06:00:00+02:00").unwrap();
        let expected = parse_iso8601_to_unix("2026-04-25T04:00:00Z").unwrap();
        assert_eq!(ts, expected);
    }

    #[test]
    fn test_unix_roundtrip() {
        let original = "2025-06-15T12:30:45Z";
        let ts = parse_iso8601_to_unix(original).unwrap();
        let back = unix_to_iso8601(ts);
        assert_eq!(back, original);
    }

    #[test]
    fn test_extract_updated_from_frontmatter() {
        let content = "---\ntitle: My Note\nupdated: 2026-04-24T23:45:37-05:00\ntags:\n  - rust\n---\n\n# Content";
        let updated = extract_updated_from_frontmatter(content).unwrap();
        assert_eq!(updated, "2026-04-24T23:45:37-05:00");
    }

    #[test]
    fn test_extract_updated_missing() {
        let content = "---\ntitle: My Note\n---\n\n# Content";
        assert!(extract_updated_from_frontmatter(content).is_none());
    }

    #[test]
    fn test_extract_updated_no_frontmatter() {
        let content = "# Just a heading\n\nNo frontmatter here.";
        assert!(extract_updated_from_frontmatter(content).is_none());
    }

    #[test]
    fn test_extract_updated_quoted() {
        let content = "---\nupdated: \"2026-04-24T23:45:37-05:00\"\n---\n";
        let updated = extract_updated_from_frontmatter(content).unwrap();
        assert_eq!(updated, "2026-04-24T23:45:37-05:00");
    }
}
