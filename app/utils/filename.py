import re

def sanitize_filename(name: str, fallback: str = "") -> str:
    """
    Sanitize a string to be safe for use as a filename.
    
    - Replaces non-alphanumeric characters (except - and .) with underscores.
    - Collapses multiple underscores into one.
    - Strips leading/trailing underscores and whitespace.
    - Truncates to 50 characters to avoid filesystem limits.
    
    Args:
        name: The input string (e.g., clip title).
        fallback: Value to return if the result is empty.
        
    Returns:
        A safe filename string, or the fallback.
    """
    if not name:
        return fallback
        
    # Replace non-alphanumeric chars (except - and .) with underscore
    clean = re.sub(r'[^a-zA-Z0-9\-\.]', '_', name)
    
    # Collapse multiple underscores
    clean = re.sub(r'_+', '_', clean)
    
    # Strip leading/trailing underscores and whitespace
    clean = clean.strip('_ ')
    
    # Truncate to reasonable length
    clean = clean[:50]
    
    # Remove trailing underscore if created by truncation
    clean = clean.rstrip('_')
    
    if not clean:
        return fallback
        
    return clean
