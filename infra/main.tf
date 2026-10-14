# Corvita ICU risk API on GCP.
#
#   bucket           raw data, processed data, model artifacts (versioned, private)
#   registry         Docker images, tags immutable, tag = model version
#   cloud run        the API; private, scales to zero
#   service accounts runtime (no GCP permissions), CI (push + deploy)
#   logs, budget     log retention and a spending alert
#
# Who can do what:
#   ML engineers     read/write the bucket (train, upload models)
#   CI               push images, deploy new revisions, run them as the runtime account;
#                    signs in from GitHub Actions with Workload Identity Federation (wif.tf),
#                    so no service account key exists anywhere
#   API runtime      nothing: the model is inside the image, logs go to stdout
#   Callers          run.invoker on the service only

data "google_project" "this" {
  project_id = var.project_id
}

locals {
  apis = [
    "run.googleapis.com",
    "artifactregistry.googleapis.com",
    "storage.googleapis.com",
    "logging.googleapis.com",
    "iam.googleapis.com",
    "billingbudgets.googleapis.com",
    "iamcredentials.googleapis.com", # short-lived tokens for Workload Identity Federation
    "sts.googleapis.com",            # exchanges GitHub's OIDC token for a Google token
  ]
  labels = {
    app     = "corvita-icu"
    managed = "terraform"
  }
}

resource "google_project_service" "apis" {
  for_each           = toset(local.apis)
  project            = var.project_id
  service            = each.value
  disable_on_destroy = false
}

# ---- storage -------------------------------------------------------------------

resource "google_storage_bucket" "ml" {
  name                        = "${var.project_id}-corvita-ml"
  location                    = var.region
  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"
  force_destroy               = false # destroy fails while data exists; empty it on purpose first
  labels                      = local.labels

  versioning {
    enabled = true
  }

  # Raw patient-level files expire. Models under models/ do not.
  lifecycle_rule {
    condition {
      age            = var.raw_data_retention_days
      matches_prefix = ["raw/"]
    }
    action {
      type = "Delete"
    }
  }

  # Old versions of any object are kept for a while for recovery, then removed.
  lifecycle_rule {
    condition {
      days_since_noncurrent_time = var.noncurrent_version_days
    }
    action {
      type = "Delete"
    }
  }

  depends_on = [google_project_service.apis]
}

# ---- image registry ---------------------------------------------------------------

resource "google_artifact_registry_repository" "images" {
  location      = var.region
  repository_id = "corvita-icu"
  format        = "DOCKER"
  description   = "API images. Tag = model version. Tags cannot be overwritten, so rollback targets stay intact."
  labels        = local.labels

  docker_config {
    immutable_tags = true
  }

  cleanup_policy_dry_run = false
  cleanup_policies {
    id     = "keep-recent"
    action = "KEEP"
    most_recent_versions {
      keep_count = 10
    }
  }

  depends_on = [google_project_service.apis]
}

# ---- service accounts and access ---------------------------------------------------

resource "google_service_account" "runtime" {
  account_id   = "corvita-api"
  display_name = "Corvita API runtime (no project roles)"
}

resource "google_service_account" "ci" {
  account_id   = "corvita-ci"
  display_name = "Corvita CI: push images, deploy revisions"
}

resource "google_artifact_registry_repository_iam_member" "ci_push" {
  location   = google_artifact_registry_repository.images.location
  repository = google_artifact_registry_repository.images.name
  role       = "roles/artifactregistry.writer"
  member     = "serviceAccount:${google_service_account.ci.email}"
}

resource "google_project_iam_member" "ci_deploy" {
  project = var.project_id
  role    = "roles/run.developer"
  member  = "serviceAccount:${google_service_account.ci.email}"
}

# CI may deploy revisions that run as the runtime account, and as nothing else.
resource "google_service_account_iam_member" "ci_acts_as_runtime" {
  service_account_id = google_service_account.runtime.name
  role               = "roles/iam.serviceAccountUser"
  member             = "serviceAccount:${google_service_account.ci.email}"
}

resource "google_storage_bucket_iam_member" "ml_engineers" {
  for_each = toset(var.ml_engineers)
  bucket   = google_storage_bucket.ml.name
  role     = "roles/storage.objectAdmin"
  member   = each.value
}

# ---- API service --------------------------------------------------------------------

resource "google_cloud_run_v2_service" "api" {
  name                = "corvita-icu-api"
  location            = var.region
  ingress             = "INGRESS_TRAFFIC_ALL" # reachable, but every call needs an identity with run.invoker
  deletion_protection = false                 # allow `terraform destroy` for this prototype
  labels              = local.labels

  template {
    service_account = google_service_account.runtime.email
    labels          = merge(local.labels, { model_version = var.model_version })

    scaling {
      min_instance_count = 0 # no traffic, no cost
      max_instance_count = var.max_instances
    }

    containers {
      image = var.image

      ports {
        container_port = 8000
      }

      env {
        name  = "MODEL_DIR"
        value = "/app/models/${var.model_version}"
      }

      resources {
        limits = {
          cpu    = "1"
          memory = "512Mi"
        }
      }

      startup_probe {
        http_get {
          path = "/health"
        }
      }

      liveness_probe {
        http_get {
          path = "/health"
        }
      }
    }
  }

  # Terraform creates the service; releases (new image, traffic split, rollback) are done by
  # .github/workflows/deploy.yml and gcloud. Ignoring these fields stops `terraform apply`
  # from rolling a release back to var.image.
  lifecycle {
    ignore_changes = [
      template[0].containers[0].image,
      template[0].labels,
      traffic,
      client,
      client_version,
    ]
  }

  depends_on = [google_project_service.apis]
}

resource "google_cloud_run_v2_service_iam_member" "invokers" {
  for_each = toset(var.invokers)
  project  = var.project_id
  location = google_cloud_run_v2_service.api.location
  name     = google_cloud_run_v2_service.api.name
  role     = "roles/run.invoker"
  member   = each.value
}

# ---- logs and cost --------------------------------------------------------------------

# Cloud Run sends stdout (our JSON request logs) to the _Default bucket.
resource "google_logging_project_bucket_config" "default" {
  project        = var.project_id
  location       = "global"
  bucket_id      = "_Default"
  retention_days = var.log_retention_days
}

resource "google_billing_budget" "monthly" {
  count           = var.billing_account == "" ? 0 : 1
  billing_account = var.billing_account
  display_name    = "corvita-icu monthly"

  budget_filter {
    projects = ["projects/${data.google_project.this.number}"]
  }

  amount {
    specified_amount {
      currency_code = var.budget_currency
      units         = tostring(var.monthly_budget)
    }
  }

  threshold_rules {
    threshold_percent = 0.5
  }
  threshold_rules {
    threshold_percent = 0.9
  }
  threshold_rules {
    threshold_percent = 1.0
  }
}
