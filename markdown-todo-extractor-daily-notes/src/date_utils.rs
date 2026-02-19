//! Date utility functions for daily notes
//!
//! Provides simple YYYY-MM-DD string parsing without external date libraries.
//! All dates use lexicographic comparison for sorting and filtering.

/// Validate a date string is in YYYY-MM-DD format
/// Returns true if valid, false otherwise
pub fn validate_date(date_str: &str) -> bool {
    if date_str.len() != 10 {
        return false;
    }

    // Check separators
    if date_str.chars().nth(4) != Some('-') || date_str.chars().nth(7) != Some('-') {
        return false;
    }

    // Parse components
    let parts: Vec<&str> = date_str.split('-').collect();
    if parts.len() != 3 {
        return false;
    }

    // Validate year (must be 4 digits)
    if parts[0].len() != 4 || !parts[0].chars().all(|c| c.is_ascii_digit()) {
        return false;
    }

    // Validate month (01-12)
    if parts[1].len() != 2 || !parts[1].chars().all(|c| c.is_ascii_digit()) {
        return false;
    }
    let month: u32 = parts[1].parse().unwrap_or(0);
    if !(1..=12).contains(&month) {
        return false;
    }

    // Validate day (01-31, basic check)
    if parts[2].len() != 2 || !parts[2].chars().all(|c| c.is_ascii_digit()) {
        return false;
    }
    let day: u32 = parts[2].parse().unwrap_or(0);
    if !(1..=31).contains(&day) {
        return false;
    }

    // Check days per month
    let max_days = days_in_month(parts[0], month);
    if day > max_days {
        return false;
    }

    true
}

/// Parse a date string into (year, month, day) components
/// Returns None if the date is invalid
pub fn parse_date(date_str: &str) -> Option<(u32, u32, u32)> {
    if !validate_date(date_str) {
        return None;
    }

    let parts: Vec<&str> = date_str.split('-').collect();
    let year: u32 = parts[0].parse().ok()?;
    let month: u32 = parts[1].parse().ok()?;
    let day: u32 = parts[2].parse().ok()?;

    Some((year, month, day))
}

/// Generate a range of dates between start and end (inclusive)
/// Returns empty vector if dates are invalid or start > end
pub fn date_range(start: &str, end: &str) -> Vec<String> {
    if !validate_date(start) || !validate_date(end) {
        return Vec::new();
    }

    // Simple lexicographic comparison works for YYYY-MM-DD format
    if start > end {
        return Vec::new();
    }

    let mut dates = Vec::new();
    let mut current = start.to_string();

    loop {
        dates.push(current.clone());

        // Check if we've reached the end
        if current == end {
            break;
        }

        // Increment date
        match increment_date(&current) {
            Some(next) => current = next,
            None => break,
        }

        // Safety check - prevent infinite loops
        if dates.len() > 3650 {
            // ~10 years max
            break;
        }
    }

    dates
}

/// Increment a date by one day
/// Returns None if date is invalid
fn increment_date(date_str: &str) -> Option<String> {
    let (year, month, day) = parse_date(date_str)?;

    let max_days = days_in_month(&year.to_string(), month);

    if day < max_days {
        // Same month, next day
        Some(format!("{:04}-{:02}-{:02}", year, month, day + 1))
    } else if month < 12 {
        // Next month, day 1
        Some(format!("{:04}-{:02}-01", year, month + 1))
    } else {
        // Next year, January 1
        Some(format!("{:04}-01-01", year + 1))
    }
}

/// Get the number of days in a month
/// Accounts for leap years
fn days_in_month(year_str: &str, month: u32) -> u32 {
    let year: u32 = year_str.parse().unwrap_or(2000);

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

/// Check if a year is a leap year
fn is_leap_year(year: u32) -> bool {
    (year.is_multiple_of(4) && !year.is_multiple_of(100)) || year.is_multiple_of(400)
}

/// Get today's date as YYYY-MM-DD string
/// Uses system time
pub fn today() -> String {
    let now = std::time::SystemTime::now();
    let duration = now
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let seconds = duration.as_secs();

    // Rough calculation - not perfectly accurate but sufficient for basic needs
    let days_since_epoch = seconds / 86400;
    let days_since_1970 = days_since_epoch as i64;

    // Calculate year, month, day (simplified algorithm)
    let mut year = 1970i64;
    let mut remaining_days = days_since_1970;

    // Add years
    loop {
        let days_in_year = if is_leap_year(year as u32) { 366 } else { 365 };
        if remaining_days < days_in_year {
            break;
        }
        remaining_days -= days_in_year;
        year += 1;
    }

    // Add months
    let mut month = 1;
    loop {
        let days = days_in_month(&year.to_string(), month) as i64;
        if remaining_days < days {
            break;
        }
        remaining_days -= days;
        month += 1;
    }

    let day = remaining_days + 1;

    format!("{:04}-{:02}-{:02}", year, month, day)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_date_valid() {
        assert!(validate_date("2025-01-20"));
        assert!(validate_date("2025-12-31"));
        assert!(validate_date("2000-02-29")); // Leap year
    }

    #[test]
    fn test_validate_date_invalid() {
        assert!(!validate_date("2025-01")); // Too short
        assert!(!validate_date("2025-01-20-extra")); // Too long
        assert!(!validate_date("2025/01/20")); // Wrong separators
        assert!(!validate_date("25-01-20")); // 2-digit year
        assert!(!validate_date("2025-13-20")); // Invalid month
        assert!(!validate_date("2025-01-32")); // Invalid day
        assert!(!validate_date("2025-02-30")); // Feb doesn't have 30 days
        assert!(!validate_date("2025-02-29")); // Not a leap year
        assert!(!validate_date("not-a-date")); // Non-numeric
    }

    #[test]
    fn test_parse_date() {
        assert_eq!(parse_date("2025-01-20"), Some((2025, 1, 20)));
        assert_eq!(parse_date("2025-12-31"), Some((2025, 12, 31)));
        assert_eq!(parse_date("invalid"), None);
    }

    #[test]
    fn test_date_range() {
        let range = date_range("2025-01-20", "2025-01-22");
        assert_eq!(range.len(), 3);
        assert_eq!(range[0], "2025-01-20");
        assert_eq!(range[1], "2025-01-21");
        assert_eq!(range[2], "2025-01-22");
    }

    #[test]
    fn test_date_range_single_day() {
        let range = date_range("2025-01-20", "2025-01-20");
        assert_eq!(range.len(), 1);
        assert_eq!(range[0], "2025-01-20");
    }

    #[test]
    fn test_date_range_invalid_start() {
        let range = date_range("invalid", "2025-01-20");
        assert!(range.is_empty());
    }

    #[test]
    fn test_date_range_invalid_end() {
        let range = date_range("2025-01-20", "invalid");
        assert!(range.is_empty());
    }

    #[test]
    fn test_date_range_start_after_end() {
        let range = date_range("2025-01-22", "2025-01-20");
        assert!(range.is_empty());
    }

    #[test]
    fn test_date_range_cross_month() {
        let range = date_range("2025-01-30", "2025-02-02");
        assert_eq!(range.len(), 4);
        assert_eq!(range[0], "2025-01-30");
        assert_eq!(range[1], "2025-01-31");
        assert_eq!(range[2], "2025-02-01");
        assert_eq!(range[3], "2025-02-02");
    }

    #[test]
    fn test_date_range_cross_year() {
        let range = date_range("2024-12-30", "2025-01-02");
        assert_eq!(range.len(), 4);
        assert_eq!(range[0], "2024-12-30");
        assert_eq!(range[1], "2024-12-31");
        assert_eq!(range[2], "2025-01-01");
        assert_eq!(range[3], "2025-01-02");
    }

    #[test]
    fn test_leap_year() {
        assert!(is_leap_year(2000));
        assert!(is_leap_year(2024));
        assert!(!is_leap_year(2025));
        assert!(!is_leap_year(1900));
    }

    #[test]
    fn test_days_in_month() {
        assert_eq!(days_in_month("2025", 1), 31); // Jan
        assert_eq!(days_in_month("2025", 2), 28); // Feb (non-leap)
        assert_eq!(days_in_month("2024", 2), 29); // Feb (leap)
        assert_eq!(days_in_month("2025", 4), 30); // April
        assert_eq!(days_in_month("2025", 12), 31); // Dec
    }
}
