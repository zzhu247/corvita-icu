output "service_url" {
  description = "API URL. Callers need run.invoker and an identity token."
  value       = google_cloud_run_v2_service.api.uri
}

output "bucket" {
  value = google_storage_bucket.ml.name
}

output "image_repository" {
  description = "Push images here, tagged with the model version."
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.images.repository_id}"
}

output "runtime_service_account" {
  value = google_service_account.runtime.email
}

output "ci_service_account" {
  value = google_service_account.ci.email
}

output "workload_identity_provider" {
  description = "Set as GitHub repo variable GCP_WIF_PROVIDER."
  value       = google_iam_workload_identity_pool_provider.github.name
}
