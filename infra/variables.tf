variable "project_id" {
  description = "GCP project that hosts everything."
  type        = string
}

variable "region" {
  description = "Region for the bucket, image registry and service. Toronto by default to keep data in Canada."
  type        = string
  default     = "northamerica-northeast2"
}

variable "image" {
  description = "Full image reference, e.g. <region>-docker.pkg.dev/<project>/corvita-icu/api:v1. The tag is the model version."
  type        = string
  default     = "us-docker.pkg.dev/cloudrun/container/hello" # placeholder until the first push
}

variable "model_version" {
  description = "Model version baked into the image. Must match the image tag."
  type        = string
  default     = "v1"
}

variable "max_instances" {
  description = "Upper bound on Cloud Run instances. Caps cost and blast radius."
  type        = number
  default     = 2
}

variable "ml_engineers" {
  description = "Members who train models and manage data, e.g. [\"group:ml-team@example.com\"]."
  type        = list(string)
  default     = []
}

variable "invokers" {
  description = "Members allowed to call the API. Never allUsers."
  type        = list(string)
  default     = []

  validation {
    condition     = !contains(var.invokers, "allUsers") && !contains(var.invokers, "allAuthenticatedUsers")
    error_message = "The API must not be public."
  }
}

variable "raw_data_retention_days" {
  description = "Objects under raw/ are deleted after this many days."
  type        = number
  default     = 90
}

variable "noncurrent_version_days" {
  description = "Overwritten or deleted object versions are kept this long, for recovery."
  type        = number
  default     = 30
}

variable "log_retention_days" {
  description = "Retention of the project's default log bucket (request logs)."
  type        = number
  default     = 30
}

variable "billing_account" {
  description = "Billing account ID for the budget alert. Empty skips the budget."
  type        = string
  default     = ""
}

variable "budget_currency" {
  description = "Must match the billing account currency."
  type        = string
  default     = "USD"
}

variable "monthly_budget" {
  description = "Monthly budget; alerts at 50%, 90% and 100%."
  type        = number
  default     = 50
}

variable "github_repository" {
  description = "owner/repo allowed to sign in as the CI service account."
  type        = string
  default     = "zzhu247/corvita-icu"
}

variable "github_deploy_branch" {
  description = "Only workflows running on this branch can sign in."
  type        = string
  default     = "main"
}
