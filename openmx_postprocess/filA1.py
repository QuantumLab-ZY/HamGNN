import sys
import string
import re

#---------------------

def clear_variables_at_level(level):
	global gvariables
	rmlist=[]
	for x in gvariables:
		if x[0]>=level:
			rmlist.append(x)
			pointer_level= x[2]+x[4]
			alloc_level=x[5]
			flag=0
			for i in range(pointer_level):
				if alloc_level[i]!=0:
					flag=1
			if flag==1:
				print "_ERROR_","some of pointer is not deallocated",x
	for x in rmlist:
		try:
			gvariables.remove(x)
		except:
			print "_SERIOUS_ERROR_","can not remove",x

#-------------------

def add_variable(indentlevel,kind,astlevel,name,blevel):
        global gvariables,glinenum,gthisfile

        for x in gvariables:
                if x[3]==name:
                        print "_WARINIG_","another definition",name, gvariables.index(x),x

        y = []
        for i in range(astlevel+blevel):
                y.append(0)
        x= [indentlevel,kind,astlevel,name,blevel,y,glinenum,gthisfile]
        print "variable definition ", x,len(gvariables)
        gvariables.append(x)

#------------------

def analyze_a_line(s):
	global gindentlevel

	i=0
        for x in s:
                if x=="=":
                        i=1
                elif x=="+":
                        i=1
                elif x=="-":
                        i=1
                elif x=="(":
                        i=1
                elif x=="{":
                        i=1
                elif x=="}":
                        i=1
                elif x=="<":
                        i=1
                elif x==">":
                        i=1
        if i==1:
                return 1

        astlevel=0
        blevel=0
        lastkey=[]
        
	if s[0]=="case":
		return 1
        if s[0]=="typedef":
                return 1
        if s[0]=="return":
                return 1
        if s[0]=="static":
                x = s[0]
                s.remove(x)
        if s[0]=="extern":
                x = s[0]
                s.remove(x)

        if s[0]=="struct":
                print "connect struct",s
                x1=s[0]
                x2=s[1]
                s.remove(x1)
                s.remove(x2)
                x12 = string.join([x1,x2],".")
                s.insert(0,x12)

        if s[0]=="unsigned":
                print "connect unsigned",s
                x1=s[0]
                x2=s[1]
                s.remove(x1)
                s.remove(x2)
                x12 =string.join([x1,x2],".")
                s.insert(0,x12)

        if len(s)<=2:
                return 1
        i=0
        
        for j in range(len(s)):
                if j==0:
                        kind=s[j]
                        lastkey=s[j]
                if s[j]=="*":
                        astlevel=astlevel+1
                elif s[j]==";":
                        add_variable(gindentlevel,kind,astlevel,name,blevel)
                        i=2
                        astlevel=0
                        blevel=0
                elif s[j]==",":
                        add_variable(gindentlevel,kind,astlevel,name,blevel)
                        i=2
                        astlevel=0
                        blevel=0
                elif s[j]=="[":
                        blevel=blevel+1
                elif s[j]=="]":
                        nothing=0
                else:
                        if blevel==0:
                                name=s[j]

        return i

#------------------------

def check_malloclevel_consistency(s):
	if s==[]:
		return
	s1 = s[1]
	s2 = s[2]
	cast_pointer=s1[1]
	sizeof_pointer=s2[1]
	if cast_pointer-1 != sizeof_pointer:
		print "_ERROR_","cast_pointer-1 != sizeof_pointer"

#---------------------------

def find_defname(name):
	global gvariables
	y=[]
	ylist=[]
	for x in gvariables:
		gname = x[3]
		if gname==name:
			y=x
			ylist.append(y)
	if len(ylist)>1:
		print "_WARNING_","more than 1 definition",name
		for z in ylist:
			print z
	if y==[]:
		print "_ERROR_","no definition",name
	return y	

#----------------------------

def check_mallocvar_consistency(s):
	global gvariables
	if s==[]:
		return

	s0=s[0]
	varname=s0[0]
	defst=find_defname(varname)
	if defst==[]:
		return
	def_pointerlevel=defst[2]+defst[4]	
	var_pointerlevel=s0[1]+s0[2]
	s1=s[1]
	cast_pointerlevel=s1[1]
	#print "pointerlevel=",def_pointerlevel,var_pointerlevel,cast_pointerlevel
	if def_pointerlevel != var_pointerlevel+cast_pointerlevel:
		print "_ERROR_","cast level",varname

	allocst=defst[5]
	if allocst[var_pointerlevel]==1:
		print "_ERROR_","pointer is already allocated",varname
	allocst[var_pointerlevel]=1
	i = gvariables.index(defst)
	defst[5]=allocst
	gvariables[i]=defst
	#print "new=",gvariables[i]

#---------------------------------------

def process_malloc(s):

	mallocst=[]

	print s
	blevel=0
	astlevel=0
	havex=0
	for x in s:
		if x=="[":
			blevel=blevel+1
		elif x=="=":
			havex=1
			break
		elif x=="*":
			astlevel=astlevel+1
		elif x=="(":
			nothing=0
		elif x==")":
			nothing=0
		else:
			if blevel==0:
				name =x
		
	if havex==0:
		return mallocst

	left = [name,astlevel,blevel]
#	print "left=",left

	mallocst.append(left)

	astlevel=0
	start=0
	bstart=0
	name=""
	for x in s:
		if start==0:
			if x=="=":
				start=1
		elif start==1:
			if x=="(":
				bstart=bstart+1
			elif x=="*":
				if bstart==1:
					astlevel=astlevel+1
			elif x==")":
				bstart=bstart-1
				if name!="" and bstart==0:
					break
			else:
				name=x
#	print "cast=",[name,astlevel]

	mallocst.append([name,astlevel])

	mallocstart=0
	sizeofstart=0
	blevel=0
	astlevel=0
	name=""
	for x in s:
		if x=="malloc":
			mallocstart=1
		if mallocstart==1:
			if x=="sizeof":
				sizeofstart=1
		if mallocstart==1 and sizeofstart==1:
			if x=="(":
				blevel=blevel+1
			elif x==")":
				blevel=blevel-1
				if blevel==0:
					break
			elif x=="*":
				astlevel=astlevel+1
			else:
				name = x
				
#	print "sizeof=",[name,astlevel]

	mallocst.append([name,astlevel])

	return  mallocst

#---------------------------------------
def process_free(s):
        global gvariables,glinenum,gthisline
        start=0
        bstart=0
        ast=0
        name=""
        blevel=0
        for x in s:
                if x=="free":
                        start=1
                elif x=="(":
                        if start==1:
                                bstart=1
                if bstart>=1:
                        if x=="*":
                                ast=ast+1
                        elif x=="[":
                                blevel=blevel+1
                        elif x=="(":
                                nothing=0
                        elif x==")":
                                nothing=0
                        else:
                                if bstart==1:
                                        name=x
                                        bstart=2

	print 1,name,ast+blevel
        status=[]
	x=find_defname(name)
	if x==[]:
		print "_ERROR_","no definition",name
		return []

	print "found",x
        status=x[5]
	def_max_pointerlevel=x[2]+x[4]
	if def_max_pointerlevel<=ast+blevel:
		print "_ERROR_","more than pointer level",s
		return []
        if status[ast+blevel]==1:
                status[ast+blevel]=0
                y=x
        elif status[ast+blevel]==0:
                print "_WARNING_ already deallocated", s
                y=x
	print "status=",status

        if status!=[]:
		print status
                i=gvariables.index(x)
                y[5]=status
                gvariables[i]=y
                print "new status=",gvariables[i]

        return [ast,name,blevel]

	

#-----------------------------------------------

def processIt(s):
	global glinenum,gthisline,gthisfile
	i = analyze_a_line(s)

	# i=2 is variable definition
	if i==2:
		return

	for x in s:
		if x=="malloc":
	#		print "malloc"
			print gthisfile,glinenum,gthisline
			mallocst=process_malloc(s)
			check_malloclevel_consistency(mallocst)
			check_mallocvar_consistency(mallocst)
		elif x=="free":
	#		print "free"
			print gthisfile,glinenum,gthisline
			freest=process_free(s)

#---------------------------------------------------------------
#           main 
#---------------------------------------------------------------

try:
	global glinenum, gthisline,gthisfile,gindentlevel,gvariables
	gindentlevel=0
	glinenum=0
	gvariables=[]
	x=[]
	stringmode=0
	gthisfile=""
	while 1:
		s=sys.stdin.readline()
		gthisline=s
		if s=="":
			print "File",gthisfile,": processed successfully"
			break
		glinenum=glinenum+1
		#print "linenum=",glinenum,s
		if s[0]=="#":
			s1=string.split(s)
			glinenum=string.atoi(s1[1])-1
			gthisfile=s1[2]
			#print "linenum,file=",glinenum,gthisfile
			continue
                s = re.sub("\{"," { ",s)
                s = re.sub("\}"," } ",s)
                s = re.sub("\("," ( ",s)
                s = re.sub("\)"," ) ",s)
                s = re.sub("="," = ",s)
                s = re.sub("\+"," + ",s)
                s = re.sub("\*"," * " ,s)
                s = re.sub("\["," [ ",s)
                s = re.sub("\]"," ] ",s)
                s = re.sub(","," , ",s)
                s = re.sub(";"," ; ", s)
                s = re.sub(">"," > ",s)
                s = re.sub("<"," < ",s)

		s1=string.split(s)

		#exception
		if len(s1)>6:
			# extern void free ( ... ) ; 
			if s1[0]=="extern" and s1[1]=="void":
				continue

		for y in s1:
			if y[0]=='\"' and stringmode==0:
				stringmode=1	
			if stringmode==1:
				sm=[]
				sm.append(y)
				stringmode=2
				if y[len(y)-1]=='\"':
					stringmode=0
					x.append(y)
			elif stringmode==2:
				sm.append(y)
				if y[len(y)-1]=='\"':
					stringmode=0
					smj = string.join(sm)
					x.append(smj)
			else: 
				x.append(y)
				if y==";":
					processIt(x)
					x=[]
				elif y=="{":
					gindentlevel=gindentlevel+1
					print "enter",gindentlevel,"at",glinenum,gthisfile,len(gvariables)
					processIt(x)
					x=[]
				elif y=="}":
					processIt(x)
					print "leave",gindentlevel,"at",glinenum,gthisfile,len(gvariables)
					clear_variables_at_level(gindentlevel)
					x=[]
					gindentlevel=gindentlevel-1

except:
	print "_SERIOUS_ERROR_ aborted at", glinenum,gthisfile

