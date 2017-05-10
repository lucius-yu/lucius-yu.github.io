---
title: Http-method-and-restful-api
categories:
  - technical posts
tags:
  - python
  - flask
  - restful
date: 2016-11-07 12:24:30 +0000
---

## Using HTTP Methods for RESTful Services
<table class="table table-striped table-bordered">
  <thead>
    <tr>
      <th>HTTP Verb</th>
      <th>CRUD</th>
      <th>Entire Collection (e.g. /customers)</th>
      <th>Specific Item (e.g. /customers/{id})</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>POST</td>
      <td>Create</td>
      <td>201 (Created), 'Location' header with link to /customers/{id} containing new ID.</td>
      <td>404 (Not Found), 409 (Conflict) if resource already exists..</td>
    </tr>
    <tr>
      <td>GET</td>
      <td>Read</td>
      <td>200 (OK), list of customers. Use pagination, sorting and filtering to navigate big lists.</td>
      <td>200 (OK), single customer. 404 (Not Found), if ID not found or invalid.</td>
    </tr>
    <tr>
      <td>PUT</td>
      <td>Update/Replace</td>
      <td>404 (Not Found), unless you want to update/replace every resource in the entire collection.</td>
      <td>200 (OK) or 204 (No Content).  404 (Not Found), if ID not found or invalid.</td>
    </tr>
    <tr>
      <td>PATCH</td>
      <td>Update/Modify</td>
      <td>404 (Not Found), unless you want to modify the collection itself.</td>
      <td>200 (OK) or 204 (No Content).  404 (Not Found), if ID not found or invalid.</td>
    </tr>
    <tr>
      <td>DELETE</td>
      <td>Delete</td>
      <td>404 (Not Found), unless you want to delete the whole collection—not often desirable.</td>
      <td>200 (OK).  404 (Not Found), if ID not found or invalid.</td>
    </tr>
  </tbody>
</table>

##
