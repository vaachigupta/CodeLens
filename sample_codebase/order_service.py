def get_order_by_id(order_id: int) -> dict:
    """Fetch an order record by its unique ID.

    Args:
        order_id: The unique identifier of the order.

    Returns:
        A dictionary containing order fields.

    Raises:
        ValueError: If order_id is not a positive integer.
    """
    if order_id <= 0:
        raise ValueError(f"order_id must be positive, got {order_id}")
    return {"id": order_id, "status": "pending"}


def cancel_order(order_id: int) -> dict:
    """Cancel an existing order.

    Args:
        order_id: The unique identifier of the order to cancel.

    Returns:
        A dictionary confirming cancellation.

    Raises:
        ValueError: If order_id is not a positive integer.
    """
    if order_id <= 0:
        raise ValueError(f"order_id must be positive, got {order_id}")
    return {"id": order_id, "status": "cancelled"}
