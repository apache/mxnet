output "secret_arn" {
  value = "${aws_secretsmanager_secret.github_credentials.id}"
}
