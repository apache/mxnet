from EmailBot import EmailBot


def lambda_handler(event, context):
    EB = EmailBot()
    EB.sendemail()
    return "Hello from Lambda"
