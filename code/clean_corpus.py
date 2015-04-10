import pspeech

if __name__ == '__main__':
    info = pspeech.read_info()
    
    for ind in range(0, len(info)):
        infile = info[ind][0]
        outfile = '../data_clean/' + infile.split('/')[-1]

        infile = open(infile, 'r').read()
        infile = infile.replace('&#39;ll', " will")
        infile = infile.replace('&#39;ve', " have")
        infile = infile.replace('don&#39;', "do ")
        infile = infile.replace('shouldn&#39;', "should ")
        infile = infile.replace('doesn&#39;', "does ")
        infile = infile.replace('&#39;', " ")
        
        infile = infile.replace('&rsquo;ll', " will")
        infile = infile.replace('&rsquo;ve', " have")
        infile = infile.replace('don&rsquo;', "do ")
        infile = infile.replace('shouldn&rsquo;', "should ")
        infile = infile.replace('doesn&rsquo;', "does ")
        infile = infile.replace('&rsquo;', " ")
        
        infile = infile.replace('&mdash;', " ")
        infile = infile.replace('&nbsp;', " ")

        infile = infile.replace(' s ', " ")
        infile = infile.replace(' m ', " am ")
        infile = infile.replace(' t ', " not ")

        outfile = open(outfile, 'wb')
        outfile.write(infile)
