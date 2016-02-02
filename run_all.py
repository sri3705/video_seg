#In the name of God



for action in actions:
	
	create_network(action)
	for video in videos[action]:
		segment = extract_segments(video)
		segments[video] = segment

	action_features = []
	for video in actions[action]
		features = segments[video].features()
		action_features.append(features)
		
	database = DB(action)
	database.save(action_features)

	open(action_datalist.txt, action)

