#include <errorHandling.hpp>
#include <stdlib.h>
#include <stdarg.h>
#include <stdio.h>

void _printf(const char *fmt, va_list ap)
{
    if (fmt) vfprintf(stderr, fmt, ap);
    //putc('\n', stderr);
}

void error(const char *fmt, ...) {
	fprintf(stderr, "ERROR ->");
	va_list ap;
	va_start(ap, fmt);
	_printf(fmt, ap);
	va_end(ap);
	exit(1);
}

void massert(bool condi, const char *fmt, ...) {
	if (!condi) {
		va_list ap;
		va_start(ap, fmt);
		_printf(fmt, ap);
		va_end(ap);
		exit(1);
  	}
}