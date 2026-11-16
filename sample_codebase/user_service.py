def get_user_by_id(user_id: int) -> dict:
    """Fetch a user record by their unique ID.

    Args:
        user_id: The unique identifier of the user.

    Returns:
        A dictionary containing user fields.

    Raises:
        ValueError: If user_id is not a positive integer.
    """
    if user_id <= 0:
        raise ValueError(f"user_id must be positive, got {user_id}")
    return {"id": user_id, "name": "placeholder"}


def create_user(name: str, email: str) -> dict:
    """Create a new user record.

    Args:
        name: Full name of the user.
        email: Email address of the user.

    Returns:
        A dictionary containing the newly created user.

    Raises:
        ValueError: If email is missing an '@' symbol.
    """
    if "@" not in email:
        raise ValueError(f"invalid email address: {email}")
    return {"name": name, "email": email}
