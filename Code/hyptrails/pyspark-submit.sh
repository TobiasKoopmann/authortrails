#!/usr/bin/env bash
# Usage: Either "./run_local authortrails".py or "./run_local authortrails"
docker build -t ls6-staff-registry.informatik.uni-wuerzburg.de/koopmann-spark/regio_spark:latest .
docker push ls6-staff-registry.informatik.uni-wuerzburg.de/koopmann-spark/regio_spark:latest
spark-submit --master k8s://https://ls6-kubebalancer.informatik.uni-wuerzburg.de:8383 \
    --deploy-mode cluster \
    --name regio-trails \
    --conf spark.jars.ivy=/tmp/.ivy \
    --conf spark.executor.instances=10 \
    --conf spark.driver.memory=6g \
    --conf spark.executor.memory=6g \
    --conf spark.executor.cores=4 \
    --conf spark.kubernetes.executor.request.cores=4 \
    --conf spark.kubernetes.executor.limit.cores=4 \
    --conf spark.kubernetes.namespace=koopmann \
    --conf spark.kubernetes.container.image.pullSecrets=ls6-staff-registry \
    --conf spark.kubernetes.container.image.pullPolicy=Always \
    --conf spark.kubernetes.pyspark.pythonVersion=3 \
    --conf spark.kubernetes.driver.volumes.persistentVolumeClaim.cephhome.mount.path=/opt/spark/work-dir/home/ \
    --conf spark.kubernetes.driver.volumes.persistentVolumeClaim.cephhome.options.claimName=home-pv-claim \
    --conf spark.kubernetes.executor.volumes.persistentVolumeClaim.cephhome.mount.path=/opt/spark/work-dir/home/ \
    --conf spark.kubernetes.executor.volumes.persistentVolumeClaim.cephhome.options.claimName=home-pv-claim \
    --conf spark.kubernetes.container.image=ls6-staff-registry.informatik.uni-wuerzburg.de/koopmann-spark/regio_spark:latest \
    /opt/spark/work-dir/authortrails.py "$1"