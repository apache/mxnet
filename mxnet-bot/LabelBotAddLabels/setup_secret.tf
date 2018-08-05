
resource "aws_secretsmanager_secret" "github_credentials"{
  name = "${var.secret_name}"
}

resource"aws_secretsmanager_secret_version" "github_credentials"{
  secret_id  = "${aws_secretsmanager_secret.github_credentials.id}"
  secret_string = "${jsonencode(var.github_credentials)}"
}
