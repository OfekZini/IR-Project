INSTANCE_NAME="instance-1"
REGION=us-central1
ZONE=us-central1-c
PROJECT_NAME="ir-hw3-444616"
IP_NAME="$PROJECT_NAME-ip"
GOOGLE_ACCOUNT_NAME="ziniofe" # without the @post.bgu.ac.il or @gmail.com part

export INSTANCE_NAME="instance-1"
export REGION=us-central1
export ZONE=us-central1-c
export PROJECT_NAME="ir-hw3-444616"
export IP_NAME="$PROJECT_NAME-ip"
export GOOGLE_ACCOUNT_NAME="ziniofe"
export LOCAL_PATH_TO=.
echo $INSTANCE_NAME
echo $REGION
echo $ZONE
echo $PROJECT_NAME
echo $IP_NAME
echo $GOOGLE_ACCOUNT_NAME


# 0. Install Cloud SDK on your local machine or using Could Shell
# check that you have a proper active account listed
gcloud auth list 
# check that the right project and zone are active
gcloud config list
# if not set them
# gcloud config set project $PROJECT_NAME
# gcloud config set compute/zone $ZONE

# 1. Set up public IP
gcloud compute addresses create $IP_NAME --project=$PROJECT_NAME --region=$REGION
gcloud compute addresses list
# note the IP address printed above, that's your extrenal IP address.
# Enter it here: 
INSTANCE_IP="34.27.12.236"

echo "2. Create Firewall rule to allow traffic to port 8080 on the instance"
# 2. Create Firewall rule to allow traffic to port 8080 on the instance
gcloud compute firewall-rules create default-allow-http-8080 \
  --allow tcp:8080 \
  --source-ranges 0.0.0.0/0 \
  --target-tags http-server
echo "3. Create the instance. Change to a larger instance (larger than e2-micro) as needed."

# # 3. Create the instance. Change to a larger instance (larger than e2-micro) as needed.
# gcloud compute instances create $INSTANCE_NAME \
#   --zone=$ZONE \
#   --machine-type=e2-standard-2 \
#   --network-interface=address=$INSTANCE_IP,network-tier=PREMIUM,subnet=default \
#   --metadata-from-file startup-script=startup_script_gcp.sh \
#   --scopes=https://www.googleapis.com/auth/cloud-platform \
#   --tags=http-server

gcloud compute instances create $INSTANCE_NAME \
  --zone=$ZONE \
  --machine-type=e2-standard-2 \
  --network-interface=address=$INSTANCE_IP,network-tier=PREMIUM,subnet=default \
  --metadata-from-file startup-script=startup_script_gcp.sh \
  --scopes=https://www.googleapis.com/auth/cloud-platform \
  --tags=http-server \
  --boot-disk-type=pd-ssd \
  --boot-disk-size=50GB

# monitor instance creation log using this command. When done (4-5 minutes) terminate using Ctrl+C
gcloud compute instances tail-serial-port-output $INSTANCE_NAME --zone $ZONE

# 4. Secure copy your app to the VM
echo "compute SCP"
gcloud compute scp $LOCAL_PATH_TO/search_frontend.py $GOOGLE_ACCOUNT_NAME@$INSTANCE_NAME:/home/$GOOGLE_ACCOUNT_NAME
gcloud compute scp $LOCAL_PATH_TO/backend.py $GOOGLE_ACCOUNT_NAME@$INSTANCE_NAME:/home/$GOOGLE_ACCOUNT_NAME
gcloud compute scp $LOCAL_PATH_TO/inverted_index_gcp.py $GOOGLE_ACCOUNT_NAME@$INSTANCE_NAME:/home/$GOOGLE_ACCOUNT_NAME
gcloud compute scp $LOCAL_PATH_TO/similarity_functions.py $GOOGLE_ACCOUNT_NAME@$INSTANCE_NAME:/home/$GOOGLE_ACCOUNT_NAME
gcloud compute scp $LOCAL_PATH_TO/parameter_search.py $GOOGLE_ACCOUNT_NAME@$INSTANCE_NAME:/home/$GOOGLE_ACCOUNT_NAME
gcloud compute scp $LOCAL_PATH_TO/queries_train.json $GOOGLE_ACCOUNT_NAME@$INSTANCE_NAME:/home/$GOOGLE_ACCOUNT_NAME
gcloud compute scp $GOOGLE_ACCOUNT_NAME@$INSTANCE_NAME:/home/$GOOGLE_ACCOUNT_NAME/detailed_results.csv $LOCAL_PATH_TO

gcloud compute scp $LOCAL_PATH_TO/detailed_results.csv $GOOGLE_ACCOUNT_NAME@$INSTANCE_NAME:/home/$GOOGLE_ACCOUNT_NAME

# 5. SSH to your VM and start the app
gcloud compute ssh $GOOGLE_ACCOUNT_NAME@$INSTANCE_NAME
python3 search_frontend.py

################################################################################
# Clean up commands to undo the above set up and avoid unnecessary charges
gcloud compute instances delete -q $INSTANCE_NAME
# make sure there are no lingering instances
gcloud compute instances list
# delete firewall rule
gcloud compute firewall-rules delete -q default-allow-http-8080
# delete external addresses
gcloud compute addresses delete -q $IP_NAME --region $REGION

echo "Finished successfully"