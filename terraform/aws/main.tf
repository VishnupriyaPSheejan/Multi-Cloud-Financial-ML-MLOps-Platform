terraform {
  required_version = ">= 1.5.0"
  required_providers {
    aws = { source = "hashicorp/aws", version = "~> 5.0" }
  }
}
provider "aws" { region = var.aws_region }
variable "aws_region" { default = "ap-southeast-1" }
resource "aws_s3_bucket" "financial_ml_data" {
  bucket = "demo-financial-ml-data-change-me"
}
# Reference architecture: S3 -> Glue -> Redshift -> SageMaker -> CloudWatch.
# This file is intentionally not required for the local demo.
