package com.Magisterka.shared.config;


import com.Magisterka.shared.domain.ErrorResponse;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.StringUtils;
import org.springframework.context.support.DefaultMessageSourceResolvable;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpStatus;
import org.springframework.http.HttpStatusCode;
import org.springframework.http.ResponseEntity;
import org.springframework.security.access.AccessDeniedException;
import org.springframework.web.bind.MethodArgumentNotValidException;
import org.springframework.web.bind.MissingRequestHeaderException;
import org.springframework.web.bind.MissingServletRequestParameterException;
import org.springframework.web.bind.ServletRequestBindingException;
import org.springframework.web.bind.annotation.ControllerAdvice;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.context.request.WebRequest;
import org.springframework.web.servlet.mvc.method.annotation.ResponseEntityExceptionHandler;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.stream.Collectors;

@Slf4j
@ControllerAdvice
public class GlobalControllerAdvice extends ResponseEntityExceptionHandler {

    private final BaseExceptionHandler baseExceptionHandler;

    public GlobalControllerAdvice(BaseExceptionHandler baseExceptionHandler) {
        this.baseExceptionHandler = baseExceptionHandler;
    }


    @Override
    protected ResponseEntity<Object> handleMethodArgumentNotValid(MethodArgumentNotValidException ex,
                                                                  HttpHeaders headers, HttpStatusCode status, WebRequest request) {
        log.error("Method arguments exception caught ", ex);

        String error = ex
                .getBindingResult()
                .getFieldErrors()
                .stream()
                .map(DefaultMessageSourceResolvable::getDefaultMessage)
                .collect(Collectors.joining(", "));

        if (StringUtils.isBlank(error)) {
            error = ex
                    .getBindingResult()
                    .getGlobalErrors()
                    .stream()
                    .map(DefaultMessageSourceResolvable::getDefaultMessage)
                    .collect(Collectors.joining(", "));
        }

        return new ResponseEntity<>(ErrorResponse.from(error,
                HttpStatus.BAD_REQUEST.value(),
                MethodArgumentNotValidException.class.getName()), HttpStatus.BAD_REQUEST);
    }

    @Override
    protected ResponseEntity<Object> handleMissingServletRequestParameter(MissingServletRequestParameterException ex,
                                                                          HttpHeaders headers, HttpStatusCode status, WebRequest request) {
        return createExceptionResponseEntity(ex, status);
    }

    @Override
    protected ResponseEntity<Object> handleServletRequestBindingException(ServletRequestBindingException ex,
                                                                          HttpHeaders headers, HttpStatusCode status, WebRequest request) {
        return createExceptionResponseEntity(ex, status);
    }

    @ExceptionHandler(AccessDeniedException.class)
    public ResponseEntity<ErrorResponse> accessDeniedExceptionHandler(AccessDeniedException exception) {
        log.error("AccessDeniedException caught ", exception);
        return baseExceptionHandler.createErrorResponse(exception, HttpStatus.UNAUTHORIZED);
    }

    @ExceptionHandler(Exception.class)
    public ResponseEntity<ErrorResponse> handleUnexpectedError(Exception exception) {
        log.error("Unexpected server error: ", exception);
        return baseExceptionHandler.createErrorResponse(exception, HttpStatus.CONFLICT);
    }

    @ExceptionHandler(MissingRequestHeaderException.class)
    public ResponseEntity<ErrorResponse> handleMissingRequestHeaderException(MissingRequestHeaderException exception) {
        log.error("Missing request header: ", exception);
        return baseExceptionHandler.createErrorResponse(exception, HttpStatus.BAD_REQUEST);
    }

    private ResponseEntity<Object> createExceptionResponseEntity(ServletRequestBindingException exception,
                                                                 HttpStatusCode status) {
        Map<String, Object> body = new LinkedHashMap<>();
        body.put("status", status);
        body.put("message", exception.getMessage());
        return new ResponseEntity<>(body, HttpStatus.BAD_REQUEST);
    }
}
