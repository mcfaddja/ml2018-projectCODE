import os
import sys
import xml.etree.ElementTree as ET
import pandas as pd


#base_path = os.path.dirname(os.path.realpath(__file__))



df = pd.read_csv(sys.argv[1]+"/profile/profile.csv")
user_list = df["userid"]

for user in user_list:

	#xml_file = os.path.join(base_path,"outputs/"+user+".xml")

	root =ET.Element ("user")

	tree = ET.ElementTree(root)

	root.set("id", user)
	root.set("age_group","xx-24")
	root.set("gender","female")
	root.set("extrovert","3.49")
	root.set("neurotic","2.73")
	root.set("agreeable","3.58")
	root.set("conscientious","3.45")
	root.set("open","3.91")

	#tree.write(xml_file)
	tree.write(sys.argv[2]+"/"+user+".xml")
