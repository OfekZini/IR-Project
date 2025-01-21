#!/bin/bash

set -euxo pipefail

readonly SPARK_JARS_DIR=/usr/lib/spark/jars
readonly SPARK_NLP_VERSION="3.2.1" # Must include subminor version here

PIP_PACKAGES=(
  "nltk==3.6.3",
  "google-cloud-storage==1.43.0"
)
readonly PIP_PACKAGES

mkdir -p ${SPARK_JARS_DIR}

function execute_with_retries() {
  local -r cmd=$1
  for ((i = 0; i < 10; i++)); do
    if eval "$cmd"; then
      return 0
    fi
    sleep 5
  done
  echo "Cmd '${cmd}' failed."
  return 1
}

function download_spark_jar() {
  local -r url=$1
  local -r jar_name=${url##*/}
  curl -fsSL --retry-connrefused --retry 10 --retry-max-time 30 \
    "${url}" -o "${SPARK_JARS_DIR}/${jar_name}"
}

function install_pip_packages() {
  execute_with_retries "pip install ${PIP_PACKAGES[*]}"
}

function install_spark_nlp() {
  download_spark_jar "https://repos.spark-packages.org/graphframes/graphframes/0.8.2-spark3.1-s_2.12/graphframes-0.8.2-spark3.1-s_2.12.jar"
}

function main() {
  # Install Spark Libraries
  echo "Installing Spark-NLP jars"
  install_spark_nlp

  # Install Pip packages
  echo "Installing Pip Packages"
  install_pip_packages
}

main
