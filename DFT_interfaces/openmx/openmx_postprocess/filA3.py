#
# Usage : 
#  /lib/cpp (-I include_directory)  filename.c | python thisfile.py 
#
#
# Copyright(C) 2006 
# Hiori Kino 
# All rights reserved. 
#
# log 
#
# filA3.py:  Jan. 11, 2006
#
#

import sys
import string
import re

#---------------------

def variables_clear_at_level(level):
	global gvariables,gthisfile,glinenum
	_name_=", in variables_clear_at_level"
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
				if x[0]==0:
					print "_WARNING_",gthisfile,glinenum,"pointer is not deallocated",x,_name_
					print
				elif x[0]%10==0:
					print "_ERROR_",gthisfile,glinenum,"some of pointer is not deallocated",x,_name_
					print
				else:
					print "_WARNING_",gthisfile,glinenum,"pointer is passed to parent function",x ,_name_
					print
	for x in rmlist:
		try:
			gvariables.remove(x)
		except:
			print "_SERIOUS_ERROR_",gthisfile,glinenum,"can not remove",x,_name_
			print

#-------------------

def variables_add_a(indentlevel,kind,astlevel,name,blevel,funcname="",iarg=0):
        global gvariables,glinenum,gthisfile
	_name_=", in variables_add_a"

	if name==")":
		return

        for x in gvariables:
                if x[3]==name:
                        print "_WARINIG_","another definition",name, gvariables.index(x),x,_name_

        y = []
        for i in range(astlevel+blevel):
                y.append(0)
        x= [indentlevel,kind,astlevel,name,blevel,y,glinenum,gthisfile,funcname,iarg]
        print "variable definition ", x,len(gvariables)
        gvariables.append(x)

#------------------

def process_function_definition(s):
	global gindentlevel,gthisfile,glinenum
	_name_=", in process_function_definition"
	if s[0]=="extern":
		return

	if gindentlevel!=10:
		return

	print "function=",s

	ipos=-1
	for i in range(len(s)):
		if s[i]=="(":
			ipos=i
			break
	if ipos<0:
		return


	ast=0
	type=""
	funcname=""
	for i in range(ipos):
		if s[i]=="static":
			continue
		elif s[i]=="*":
			ast=ast+1
		else:
			if type=="":
				type=s[i]
			else:
				funcname=s[i]

#	print "type,ast,funcname=",type,"-",ast,"-",funcname
	if type!="" and funcname=="":
		funcname=type
		type="void"
	if type=="" or funcname=="":
		print "_SERIOUS_ERROR_",gthisfile,glinenum,"error in process_function_definition 1",_name_
		print
		abort(0)

	arglist=[]
	arg=[]
	for  i in range(len(s)):
		if i<=ipos:
			continue
		if s[i]==")":
			continue
		if s[i]=="{":
			continue
		if s[i]==",":
			if arg!=[]:
				arglist.append(arg)
				arg=[]
				continue
		arg.append(s[i])
	if arg!=[]:
		arglist.append(arg)
		
	print "arglist=",arglist	

	iarg=0
	for x in arglist:
		type=""
		name=""
		ast=0
		blevel=0
		blevel0=0
		for y in x:
			if y=="*":
				ast=ast+1
			if y=="[":
				blevel=blevel+1
				blevel0=blevel0+1
			if y=="]":
				blevel=blevel-1
			else:
				if blevel==0:
					if type=="":
						type=y
					else:
						name=y
		if type=="void" and name=="":
			continue
		if type=="" or name=="":
			print "_SERIOUS_ERROR","error in process_function_definition 2",_name
			print
			abort(2)
		variables_add_a(gindentlevel+1,type,ast,name,blevel0,funcname,iarg)
		iarg=iarg+1
	
	

#------------------

def process_analyze_eq(s):
	global gthisfile, gthisline,glinenum
	_name_=", in process_analyze_eq"
	if len(s)==1:
		return
	for i in range(len(s)-1):
		if s[i]=="=" and s[i+1]=="+":
			print s
			print gthisfile, glinenum,gthisline
			print "_WARNING_",gthisfile,glinenum,"possible += ?"
			print
		if s[i]=="=" and s[i+1]=="-":
			print s
			print gthisfile, glinenum,gthisline
	                print "_WARNING_",gthisfile,glinenum,"possible -= ?"
			print

#-----------------------

def process_vardef(s):
	global gindentlevel


	if s[0]=="if":
		return 1
	if s[0]=="for":
		return 1
	if s[0]=="else":
		return 1

	i=0
	for x in s:
		if x=="(":
			i=1
		elif x==")":
			if i==1:
				i=2
		elif x=="{":
			if i==2:
				i=3
	if i==3:
		# this is a function definition, maybe
		process_function_definition(s)
		return 2

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
                        variables_add_a(gindentlevel,kind,astlevel,name,blevel)
                        i=2
                        astlevel=0
                        blevel=0
                elif s[j]==",":
                        variables_add_a(gindentlevel,kind,astlevel,name,blevel)
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

def malloc_check_pointerlevel_consistency(s):
	global gthisfile,glinenum
	if s==[]:
		return
	s1 = s[1]
	s2 = s[2]
	cast_pointer=s1[1]
	sizeof_pointer=s2[1]
	if cast_pointer-1 != sizeof_pointer:
	#	print gthisfile,glinenum,gthisline
		print "_ERROR_",gthisfile,glinenum,"cast_pointer-1 != sizeof_pointer"
		print

#---------------------------

def variables_find_defname(name):
	global gvariables,gthisfile,glinenum
	_name_=", in variables_find_defname"
	y=[]
	ylist=[]
	for x in gvariables:
		gname = x[3]
		if gname==name:
			y=x
			ylist.append(y)
	if len(ylist)>1:
		print "_WARNING_",gthisfile,glinenum,"more than 1 definition",name
		print
		for z in ylist:
			print z
	if y==[]:
		print "_ERROR_",gthisfile,glinenum,"no definition",name,_name_
		print

	return y	

#----------------------------

def malloc_check_var_consistency(s):
	global gvariables,gthisfile,glinenum
	_name_=", in malloc_check_var_consistency"
	if s==[]:
		return

	s0=s[0]
	varname=s0[0]
	defst=variables_find_defname(varname)
	if defst==[]:
		return
	def_pointerlevel=defst[2]+defst[4]	
	var_pointerlevel=s0[1]+s0[2]
	s1=s[1]
	cast_pointerlevel=s1[1]
	#print "pointerlevel=",def_pointerlevel,var_pointerlevel,cast_pointerlevel
	if def_pointerlevel != var_pointerlevel+cast_pointerlevel:
		print "_ERROR_",gthisfile,glinenum,"ecast level",varname,_name_
		print

	allocst=defst[5]
	if allocst[var_pointerlevel]==1:
		print "_ERROR_",gthisfile,glinenum,"pointer is already allocated",varname,_name_
		print
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
        global gvariables,glinenum,gthisline,gthisfile
	_name_=", in process_free"
	print "s=",s
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

	#print 1,name,ast+blevel
        status=[]
	x=variables_find_defname(name)
	if x==[]:
		print "_ERROR_",gthisfile,glinenum,"no definition",name,_name_
		print
		return []

	#print "found",x
        status=x[5]
	def_max_pointerlevel=x[2]+x[4]
	if def_max_pointerlevel<=ast+blevel:
		print "_ERROR_",gthisfile,glinenum,"more than pointer level",s,_name_
		print
		return []
        if status[ast+blevel]==1:
                status[ast+blevel]=0
                y=x
        elif status[ast+blevel]==0:
                print "_ERROR_",gthisfile,glinenum, "already deallocated", s,_name_
		print
                y=x
	#print "status=",status

        if status!=[]:
	#	print status
                i=gvariables.index(x)
                y[5]=status
                gvariables[i]=y
                #print "new status=",gvariables[i]

        return [ast,name,blevel]

	

#-----------------------------------------------

def processIt(s):
	global glinenum,gthisline,gthisfile

	process_analyze_eq(s)

	i = process_vardef(s)

	# i=2 is variable definition
	if i==2:
		return

	for x in s:
		if x=="malloc":
	#		print "malloc"
			print gthisfile,glinenum,gthisline
			mallocst=process_malloc(s)
			malloc_check_pointerlevel_consistency(mallocst)
			malloc_check_var_consistency(mallocst)
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
		gthisline=re.sub('\n','',s)
		if s=="":
			print "leave",0,"at",glinenum,gthisfile,len(gvariables)
			variables_clear_at_level(0)
			print "File",gthisfile,": processed successfully"
			break
		glinenum=glinenum+1
		#print "linenum=",glinenum,s
		if s[0]=="#":
			s1=string.split(s)
			#glinenum=string.atoi(s1[1])-1
			glinenum=int(s1[1])-1
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
				if len(y)>1 and y[len(y)-1]=='\"':
					stringmode=0
					x.append(y)
					sm=[]
				else:
					stringmode=2
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
					gindentlevel=gindentlevel+10
					#print "enter",gindentlevel,"at",glinenum,gthisfile,len(gvariables)
					processIt(x)
					x=[]
				elif y=="}":
					processIt(x)
					#print "leave",gindentlevel,"at",glinenum,gthisfile,len(gvariables)
					variables_clear_at_level(gindentlevel)
					x=[]
					gindentlevel=gindentlevel-10

except:
	_name_=", in main"
	print "_SERIOUS_ERROR_ aborted at", glinenum,gthisfile,_name_

