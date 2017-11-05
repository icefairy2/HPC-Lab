#include <stdio.h>
#include <unistd.h>

int main(void)
{
	char hostname[100];
	hostname[99] = 0;

	gethostname(hostname, 99);
	printf("Hostname %s\n", hostname);
	return 0;
}

