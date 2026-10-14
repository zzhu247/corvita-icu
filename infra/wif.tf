# Workload Identity Federation: GitHub Actions signs in to Google Cloud without a key file.
#
# How it works:
#   1. A workflow with `permissions: id-token: write` gets a short-lived OIDC token from GitHub.
#      The token states the repository, branch, workflow and commit.
#   2. Google's Security Token Service checks it against the provider below.
#      attribute_condition rejects every token not from var.github_repository on
#      var.github_deploy_branch: forks, other branches and pull requests cannot sign in.
#   3. The token is exchanged for a 1-hour access token for the CI service account.
#
# Nothing long-lived exists to leak, rotate or revoke. Removing access = deleting the binding.

resource "google_iam_workload_identity_pool" "github" {
  workload_identity_pool_id = "github"
  display_name              = "GitHub Actions"
  description               = "OIDC identities from GitHub Actions"

  depends_on = [google_project_service.apis]
}

resource "google_iam_workload_identity_pool_provider" "github" {
  workload_identity_pool_id          = google_iam_workload_identity_pool.github.workload_identity_pool_id
  workload_identity_pool_provider_id = "github-actions"
  display_name                       = "GitHub Actions OIDC"

  attribute_mapping = {
    "google.subject"       = "assertion.sub"
    "attribute.repository" = "assertion.repository"
    "attribute.ref"        = "assertion.ref"
  }

  attribute_condition = "assertion.repository == \"${var.github_repository}\" && assertion.ref == \"refs/heads/${var.github_deploy_branch}\""

  oidc {
    issuer_uri = "https://token.actions.githubusercontent.com"
  }
}

# Identities from that repository may act as the CI service account, and as nothing else.
resource "google_service_account_iam_member" "ci_from_github" {
  service_account_id = google_service_account.ci.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "principalSet://iam.googleapis.com/${google_iam_workload_identity_pool.github.name}/attribute.repository/${var.github_repository}"
}
