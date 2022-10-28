# 14-813 Course Project 2 – MQTT Dataset

This project involves using the MQTT dataset to conduct data processing and machine learning. This dataset, also known as Message Queuing Telemetry Transport dataset, focuses on predicting the cyber-attacks conducted on an automated house system. Each aspect of the home, from temperature to humidity, is controlled by a sensor under the Internet of Things system and is vulnerable to cyber-attacks from malicious nodes.
This is an extremely large dataset, containing 34 features and 20 million records. Since our computer cannot manage the full volume of the dataset, a small subset of the data, containing 200,000 records was used for the purposes of this assignment. This smaller dataset contains train and test records in the same ratio as the full dataset. A summary of all the columns and the constraints is shown in the table below. 

|Feature	|Constraints	|Description|
|---------|:-------------|:---------------|
|tcp.flags	|Must be string	|TCP flags|
|tcp.time_delta	|Cannot be negative	|Time TCP stream|
|tcp.len	|Cannot be negative	|TCP Segment Len|
|mqtt.conack.flags	|-	|Acknowledge Flags|
|mqtt.conack.flags.reserved	|-	|Reserved|
|mqtt.conack.flags.sp	|-	|Session Present|
|mqtt.conack.val	|-	|Return Code|
|mqtt.conflag.cleansess	|-	|Clean Session Flag|
|mqtt.conflag.passwd	|-	|Password Flag|
|mqtt.conflag.qos	|-	|QoS Level|
|mqtt.conflag.reserved	|-	|(Reserved)|
|mqtt.conflag.retain	|-	|Will Retain|
|mqtt.conflag.uname	|-	|Username Flag|
|mqtt.conflag.willflag	|-	|Will Flag|
|mqtt.conflags	|-	|Connect Flags|
|mqtt.dupflag	|-	|DUP Flag|
|mqtt.hdrflags	|-	|Header Flags|
|mqtt.kalive	|-	|Keep Alive|
|mqtt.len	|Cannot be negative	|Msg Len|
|mqtt.msg	|Unique	|Message|
|mqtt.msgid	|Unique	|Message Identifier|
|mqtt.msgtype	|-	|Message Type|
|mqtt.proto_len	|Cannot be negative	|Protocol Name Length|
|mqtt.protoname	|-	|Protocol Name|
|mqtt.qos	|-	|QoS Level|
|mqtt.retain	|-	|Retain|
|mqtt.sub.qos	|-	|Requested QoS|
|mqtt.suback.qos	|-	|Granted QoS|
|mqtt.ver	|-	|Version|
|mqtt.willmsg	|Unique	|Will Message|
|mqtt.willmsg_len	|Cannot be negative	|Will Message Length|
|mqtt.willtopic	|Unique	|Will Topic|
|mqtt.willtopic_len	|Cannot be negative	|Will Topic Length|
|target	|Cannot be null	|Type of Attack|
train_test	|Can only be 0 or 1	|Train and Test Data|
