terraform {
  required_version = ">= 1.6.0, < 2.0.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 6.0"
    }
  }

  # Remote state. The state bucket is created once by hand (versioned, not public),
  # outside this config, so `terraform destroy` can never delete its own state.
  # Uncomment for a real deployment:
  # backend "gcs" {
  #   bucket = "<org>-terraform-state"
  #   prefix = "corvita-icu"
  # }
}

provider "google" {
  project = var.project_id
  region  = var.region
}
