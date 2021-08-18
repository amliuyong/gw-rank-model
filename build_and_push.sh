#!/bin/bash
#%%sh

# The name of our algorithm
algorithm_name=sagemaker-recsys-graph-inference

chmod +x kggraph/train
chmod +x kggraph/serve

AWS_CMD="aws"
if [[ -n $PROFILE ]]; then
  AWS_CMD="aws --profile $PROFILE"
fi

account=$(${AWS_CMD} sts get-caller-identity --query Account --output text)

# Get the region defined in the current configuration (default to us-west-2 if none defined)
region=$(${AWS_CMD} configure get region)
region=${region:-us-west-1}

fullname="${account}.dkr.ecr.${region}.amazonaws.com/${algorithm_name}:latest"

echo "ECR: ${fullname}"

sleep 3

# If the repository doesn't exist in ECR, create it.

${AWS_CMD} ecr describe-repositories --repository-names "${algorithm_name}" > /dev/null 2>&1

if [ $? -ne 0 ]
then
    ${AWS_CMD} ecr create-repository --repository-name "${algorithm_name}" > /dev/null
fi

if [[ $region =~ ^cn.* ]]
then
    registry_id="727897471807"
    registry_uri="${registry_id}.dkr.ecr.${region}.amazonaws.com.cn"
else
    registry_id="763104351884"
    registry_uri="${registry_id}.dkr.ecr.${region}.amazonaws.com"
fi

echo "registry_uri: $registry_uri"

# Get the login command from ECR and execute it directly
#$(aws ecr get-login --region ${region} --no-include-email)
#${AWS_CMD} ecr get-login --region ${region} --no-include-email

$AWS_CMD ecr get-login-password  --region ${region} | docker login --username AWS --password-stdin ${registry_id}


# Build the docker image locally with the image name and then push it to ECR
# with the full name.

rm -rf info > /dev/null 2>&1
mkdir info

cd info/

$AWS_CMD s3 cp s3://aws-gcr-rs-sol-dev-ap-southeast-1-522244679887/sample-data-news-gw-phase2/notification/action-model/model.tar.gz  ./ \
&& $AWS_CMD s3 cp s3://aws-gcr-rs-sol-dev-ap-southeast-1-522244679887/sample-data-news-gw-phase2/notification/embeddings/dkn_context_embedding.npy ./ \
&& $AWS_CMD s3 cp s3://aws-gcr-rs-sol-dev-ap-southeast-1-522244679887/sample-data-news-gw-phase2/notification/embeddings/dkn_entity_embedding.npy ./ \
&& $AWS_CMD s3 cp s3://aws-gcr-rs-sol-dev-ap-southeast-1-522244679887/sample-data-news-gw-phase2/notification/embeddings/dkn_word_embedding.npy ./

if [[ $? != 0 ]]; then
  wget https://aws-gcr-rs-sol-dev-ap-southeast-1-522244679887.s3.ap-southeast-1.amazonaws.com/sample-data-news-gw-phase2/notification/action-model/model.tar.gz
  wget https://aws-gcr-rs-sol-dev-ap-southeast-1-522244679887.s3.ap-southeast-1.amazonaws.com/sample-data-news-gw-phase2/notification/embeddings/dkn_context_embedding.npy
  wget https://aws-gcr-rs-sol-dev-ap-southeast-1-522244679887.s3.ap-southeast-1.amazonaws.com/sample-data-news-gw-phase2/notification/embeddings/dkn_entity_embedding.npy
  wget https://aws-gcr-rs-sol-dev-ap-southeast-1-522244679887.s3.ap-southeast-1.amazonaws.com/sample-data-news-gw-phase2/notification/embeddings/dkn_word_embedding.npy
fi

cd ..

#$AWS_CMD s3 cp s3://aws-gcr-rs-sol-dev-ap-southeast-1-522244679887/sample-data-news-gw-phase2/notification/inverted-list/news_id_news_feature_dict.pickle ./info/

docker build  -t ${algorithm_name} . --build-arg REGISTRY_URI=${registry_uri}
docker tag ${algorithm_name}:latest ${fullname}

rm -rf ./info

#docker push ${fullname}