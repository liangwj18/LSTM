#include <cstdio>
#include <cstdlib>
#include <string>
#include <sstream>

using namespace std;

int main()
{
	for (int i=1;i<=100;++i)
	{
		stringstream stream;  
        stream<<i;  
		string s="";
		stream >> s;
	
		string st="ffprobe -select_streams v -show_frames -show_entries frame=key_frame -of csv "+s+".mp4 > output"+s+".txt";
    	system(st.c_str());
	}
    return 0;
}
