def getPaymentById(paymentId):
    if paymentId <= 0:
        raise Exception("bad id")
    return {"id": paymentId, "status": "unpaid"}
