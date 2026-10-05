terraform {
  required_version = ">= 1.5.0"
  required_providers {
    google = { source = "hashicorp/google", version = "~> 6.0" }
  }
}
variable "project_id" {}
variable "region" { default = "asia-southeast1" }
provider "google" { project = var.project_id region = var.region }
resource "google_storage_bucket" "financial_ml_data" {
  name = "demo-financial-ml-data-change-me"
  location = "ASIA"
}
# Reference architecture: GCS -> BigQuery -> Vertex AI -> Cloud Monitoring.
# This file is intentionally not required for the local demo.
