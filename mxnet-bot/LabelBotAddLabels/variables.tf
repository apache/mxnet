variable "github_credentials" {
  default = {
    github_user = ""
    github_oauth_token  = ""
  }
  type = "map"
}

variable "secret_name" {
  default = ""
}
