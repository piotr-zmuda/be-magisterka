package com.Magisterka.shared.config;

import com.Magisterka.shared.domain.ErrorResponse;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.ControllerAdvice;

@ControllerAdvice
public class BaseExceptionHandler {

    private static final Logger log = LoggerFactory.getLogger(BaseExceptionHandler.class);

    protected ResponseEntity<ErrorResponse> createErrorResponse(Exception exception, HttpStatus status) {
        log.error("{} {} {}", exception.getClass().getSimpleName(), exception.getMessage(), exception.getStackTrace(), exception);

        return new ResponseEntity<>(ErrorResponse.from(
                exception.getMessage(),
                status.value(),
                exception.getClass().getSimpleName()
        ), status);
    }
}

