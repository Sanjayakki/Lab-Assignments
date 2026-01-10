S=input("Enter a string:")
vow="aeiouAEIOU"
count=0
rem=0
for char in S:
    if char in vow:
        count=count+1
print("Number of vowels in the string:",count)        
rem=len(S)-count
print("number of consonants in the string:",rem)

        