columns = [ 'title', 
       'upvotes', 'views', 'reads', 'firstPublishedAt', 'firstPublishedAtBucket', 'readingTime', 'claps',
       'internalReferrerViews', 'friendsLinkViews','postId','slug','collectionId','previewImage']

formatted_columns =  [ 'title', 
       'upvotes', 'views', 'reads', 'first Published At','first Published At Bucket', 'reading Length (minutes)', 'claps',
       'internal Referrer Views', 'friends Link Views','post Id','title slug','collection Id','Image']

ignore = ['month','year','first Published At','first Published At Bucket','days since published','url','publication','title','title tokens','image Url']

metric_dict = {'views':[1000,10000,100000],'read_%':[25,50,75],'claps':[100,500,1000],'reading Length (minutes)':[5,10,15]}


