use actix_web::{
    dev::{forward_ready, Service, ServiceRequest, ServiceResponse, Transform},
    Error,
};
use futures_util::future::LocalBoxFuture;
use std::{
    future::{ready, Ready},
    time::Instant,
};
use tracing::{info, error};

pub struct RequestLogging;

impl<S, B> Transform<S, ServiceRequest> for RequestLogging
where
    S: Service<ServiceRequest, Response = ServiceResponse<B>, Error = Error>,
    S::Future: 'static,
    B: 'static,
{
    type Response = ServiceResponse<B>;
    type Error = Error;
    type InitError = ();
    type Transform = RequestLoggingMiddleware<S>;
    type Future = Ready<Result<Self::Transform, Self::InitError>>;

    fn new_transform(&self, service: S) -> Self::Future {
        ready(Ok(RequestLoggingMiddleware { service }))
    }
}

pub struct RequestLoggingMiddleware<S> {
    service: S,
}

impl<S, B> Service<ServiceRequest> for RequestLoggingMiddleware<S>
where
    S: Service<ServiceRequest, Response = ServiceResponse<B>, Error = Error>,
    S::Future: 'static,
    B: 'static,
{
    type Response = ServiceResponse<B>;
    type Error = Error;
    type Future = LocalBoxFuture<'static, Result<Self::Response, Self::Error>>;

    forward_ready!(service);

    fn call(&self, req: ServiceRequest) -> Self::Future {
        let start_time = Instant::now();
        let method = req.method().to_string();
        let uri = req.uri().to_string();
        let remote_addr = req.connection_info().realip_remote_addr().unwrap_or("unknown").to_string();
        
        info!(
            method = %method,
            uri = %uri,
            remote_addr = %remote_addr,
            "Request started"
        );

        let fut = self.service.call(req);

        Box::pin(async move {
            let result = fut.await;
            let duration = start_time.elapsed();

            match &result {
                Ok(response) => {
                    let status = response.status();
                    info!(
                        method = %method,
                        uri = %uri,
                        remote_addr = %remote_addr,
                        status = %status.as_u16(),
                        duration_ms = %duration.as_millis(),
                        "Request completed"
                    );
                }
                Err(err) => {
                    error!(
                        method = %method,
                        uri = %uri,
                        remote_addr = %remote_addr,
                        duration_ms = %duration.as_millis(),
                        error = %err,
                        "Request failed"
                    );
                }
            }

            result
        })
    }
}